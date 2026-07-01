use std::{
    collections::HashSet,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use burn_automata::{
    AutomataPreset, BpkModelManifest, EquivarianceMode, MorphogenSeedEnvelope, NpaConfig, NpaModel,
    NpaWeights, ParticleSeed, RenderLossConfig, RolloutBatchConfig, RolloutConfig,
    RolloutSupervisionConfig, SgdConfig, SupervisedBatch, SupervisedTarget, TrainingRunConfig,
    feature_supervised_batch,
    kernels::build_hashgrid,
    mesh_multiview_render_loss_from_trace,
    rollout::{
        GROWTH_3D_ACTIVE_OPACITY_LOGIT, GROWTH_3D_INACTIVE_OPACITY_LOGIT,
        UV_TORUS_INITIAL_OPACITY_LOGIT, UV_TORUS_INITIAL_SCALE, UV_TORUS_MINOR_RATIO,
        UV_TORUS_NORMAL_STATE_OFFSET, UV_TORUS_OPACITY_GROWTH_DELTA,
        UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET, growth_3d_active_core_radius,
        growth_3d_domain_radius, growth_3d_material_opacity_channel, growth_3d_seed_radius,
        morphogen_seed_envelope_position, seed_particles_scaled,
        uv_torus_continuous_surface_position, uv_torus_continuous_volume_position,
        uv_torus_dense_seed_radius, uv_torus_orientation_state_available, uv_torus_outer_radius,
        uv_torus_outward_normal, uv_torus_position_color, uv_torus_project_position,
        uv_torus_sample, uv_torus_signed_distance, uv_torus_surface_error,
        uv_torus_tail_state_to_rgb,
    },
    rollout_supervised_batch, rollout_supervised_batch_from_model, run_rollout,
    run_supervised_training, supervised_backward, supervised_loss, supervised_train_step,
    target_geometry::{TriangleMeshTarget, dot3},
};
use rand::{SeedableRng, rngs::StdRng};

const CATALOG_3D_GROWTH_SEED: u64 = 0x51a7_3d;

#[test]
fn rollout_runs_with_seeded_2d_model() {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
    let model = NpaModel::seeded(config.clone(), 7);
    let trace = run_rollout(
        &model,
        &grid,
        &RolloutConfig {
            particle_count: 16,
            steps: 3,
            update_prob: 1.0,
            ..RolloutConfig::default()
        },
        ParticleSeed::UniformCircle,
    )
    .unwrap();

    assert_eq!(trace.positions.len(), 16);
    assert_eq!(trace.states.len(), 16 * config.state_dims);
    assert_eq!(trace.mean_dx.len(), 3);
    assert!(trace.mean_dx.iter().all(|v| v.is_finite()));
}

#[test]
fn supervised_step_reduces_simple_zero_model_loss() {
    let config = NpaConfig {
        hidden_dims: 4,
        ..NpaConfig::growing_2d()
    };
    let mut model = NpaModel {
        config: config.clone(),
        weights: NpaWeights::zeros(&config),
    };
    let rows = 4;
    let batch = SupervisedBatch {
        features: vec![1.0; rows * config.perception_dims()],
        target_update: vec![0.2; rows * config.update_dims()],
    };
    let before = mse(&model, &batch);
    let report = supervised_train_step(
        &mut model,
        &batch,
        SgdConfig {
            learning_rate: 0.1,
            grad_clip_norm: 100.0,
            ..SgdConfig::default()
        },
    )
    .unwrap();
    let after = mse(&model, &batch);

    assert_eq!(report.rows, rows);
    assert!(report.loss.is_finite());
    assert!(
        after < before,
        "after {after} should be less than before {before}"
    );
}

#[test]
fn supervised_backward_returns_feature_and_weight_gradients() {
    let config = NpaConfig {
        hidden_dims: 4,
        ..NpaConfig::growing_2d()
    };
    let model = NpaModel::seeded(config.clone(), 13);
    let rows = 2;
    let batch = SupervisedBatch {
        features: vec![0.25; rows * config.perception_dims()],
        target_update: vec![0.0; rows * config.update_dims()],
    };

    let (grads, report) = supervised_backward(&model, &batch).unwrap();

    assert_eq!(report.rows, rows);
    assert!(report.loss.is_finite());
    assert!(report.grad_norm.is_finite());
    assert_eq!(grads.w1.len(), model.weights.w1.len());
    assert_eq!(grads.b1.len(), model.weights.b1.len());
    assert_eq!(grads.w2.len(), model.weights.w2.len());
    assert_eq!(grads.b2.len(), model.weights.b2.len());
    assert_eq!(grads.features.len(), batch.features.len());
    assert!(grads.features.iter().all(|v| v.is_finite()));
}

#[test]
fn supervised_training_report_tracks_convergence() {
    let config = NpaConfig {
        hidden_dims: 4,
        ..NpaConfig::growing_2d()
    };
    let mut model = NpaModel {
        config: config.clone(),
        weights: NpaWeights::zeros(&config),
    };
    let rows = 8;
    let batch = SupervisedBatch {
        features: vec![0.5; rows * config.perception_dims()],
        target_update: vec![0.15; rows * config.update_dims()],
    };
    let initial = supervised_loss(&model, &batch).unwrap();
    let report = run_supervised_training(
        &mut model,
        &batch,
        TrainingRunConfig {
            steps: 16,
            report_interval: 4,
            sgd: SgdConfig {
                learning_rate: 0.05,
                grad_clip_norm: 100.0,
                ..SgdConfig::default()
            },
        },
    )
    .unwrap();

    assert_eq!(report.steps, 16);
    assert_eq!(report.rows, rows);
    assert_eq!(report.history.len(), 4);
    assert!((report.initial_loss - initial).abs() < 1.0e-6);
    assert!(
        report.final_loss < report.initial_loss,
        "final {} should be less than initial {}",
        report.final_loss,
        report.initial_loss
    );
    assert!(report.best_loss <= report.final_loss + 1.0e-6);
    assert!(report.history.iter().all(|entry| entry.loss.is_finite()));
}

#[test]
fn feature_supervised_batch_supports_2d_and_3d_teacher_training() {
    for preset in [AutomataPreset::Growing2d, AutomataPreset::Growing3dGs] {
        let (mut config, _grid) = NpaConfig::for_preset(preset);
        config.hidden_dims = 8;
        let teacher = NpaModel::seeded(config.clone(), 11);
        let mut student = NpaModel::seeded(config, 7);
        let batch = feature_supervised_batch(
            &student,
            SupervisedTarget::Teacher(&teacher),
            burn_automata::FeatureBatchConfig {
                rows: 32,
                seed: 3,
                amplitude: 0.25,
            },
        )
        .unwrap();
        let before = supervised_loss(&student, &batch).unwrap();
        let report = run_supervised_training(
            &mut student,
            &batch,
            TrainingRunConfig {
                steps: 24,
                report_interval: 24,
                sgd: SgdConfig {
                    learning_rate: 5.0e-3,
                    grad_clip_norm: 10.0,
                    ..SgdConfig::default()
                },
            },
        )
        .unwrap();

        assert_eq!(report.rows, 32);
        assert!(
            report.final_loss < before,
            "{preset:?}: final {} should be less than initial {before}",
            report.final_loss
        );
    }
}

#[test]
fn rollout_supervised_batch_supports_2d_and_3d_perception_features() {
    for preset in [AutomataPreset::Texture2d, AutomataPreset::Growing3dGs] {
        let (mut config, grid) = NpaConfig::for_preset(preset);
        config.hidden_dims = 8;
        let model = NpaModel::seeded(config.clone(), 17);
        let trace = run_rollout(
            &model,
            &grid,
            &RolloutConfig {
                particle_count: 48,
                steps: 1,
                update_prob: 1.0,
                seed_scale: NpaConfig::seed_scale_for_preset(preset),
                ..RolloutConfig::default()
            },
            ParticleSeed::UniformCircle,
        )
        .unwrap();
        let batch = rollout_supervised_batch(
            &model,
            &grid,
            &trace,
            SupervisedTarget::ZeroUpdate,
            RolloutBatchConfig {
                max_rows: 16,
                dt: 1.0,
            },
        )
        .unwrap();

        assert_eq!(batch.features.len(), 16 * config.perception_dims());
        assert_eq!(batch.target_update.len(), 16 * config.update_dims());
        assert!(batch.features.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn rollout_supervision_trains_from_local_2d_and_3d_rollout_states() {
    for preset in [AutomataPreset::Growing2d, AutomataPreset::Growing3dGs] {
        let (mut config, grid) = NpaConfig::for_preset(preset);
        config.hidden_dims = 8;
        let teacher = NpaModel::seeded(config.clone(), 31);
        let mut student = NpaModel::seeded(config.clone(), 7);
        let seed_mode = if preset == AutomataPreset::Growing3dGs {
            ParticleSeed::TorusMorphogenDense3d
        } else {
            ParticleSeed::UniformCircle
        };
        let seed_scale = if preset == AutomataPreset::Growing3dGs {
            0.72
        } else {
            NpaConfig::seed_scale_for_preset(preset)
        };
        let batch = rollout_supervised_batch_from_model(
            &student,
            &teacher,
            &grid,
            SupervisedTarget::Teacher(&teacher),
            RolloutSupervisionConfig {
                max_rows: 24,
                particle_count: 32,
                rollout_steps: 2,
                rollouts: 2,
                seed: 99,
                seed_scale,
                seed_mode,
                ..RolloutSupervisionConfig::default()
            },
        )
        .unwrap();
        let before = supervised_loss(&student, &batch).unwrap();
        let report = run_supervised_training(
            &mut student,
            &batch,
            TrainingRunConfig {
                steps: 12,
                report_interval: 12,
                sgd: SgdConfig {
                    learning_rate: 1.0e-2,
                    grad_clip_norm: 10.0,
                    ..SgdConfig::default()
                },
            },
        )
        .unwrap();

        assert_eq!(report.rows, 24);
        assert!(
            report.final_loss < before,
            "{preset:?}: rollout-local final {} should be less than initial {before}",
            report.final_loss
        );
    }
}

#[test]
fn supervised_backward_matches_finite_difference_gradients() {
    let config = NpaConfig {
        hidden_dims: 2,
        ..NpaConfig::growing_2d()
    };
    let mut weights = NpaWeights::zeros(&config);
    for (idx, value) in weights.w1.iter_mut().enumerate() {
        *value = 0.01 + idx as f32 * 0.0007;
    }
    weights.b1.fill(0.35);
    for (idx, value) in weights.w2.iter_mut().enumerate() {
        *value = -0.02 + idx as f32 * 0.0013;
    }
    weights.b2.fill(0.04);
    let model = NpaModel {
        config: config.clone(),
        weights,
    };
    let rows = 3;
    let features = (0..rows * config.perception_dims())
        .map(|idx| 0.15 + (idx as f32 * 0.017).sin() * 0.03)
        .collect::<Vec<_>>();
    let target_update = (0..rows * config.update_dims())
        .map(|idx| -0.04 + (idx as f32 * 0.021).cos() * 0.02)
        .collect::<Vec<_>>();
    let batch = SupervisedBatch {
        features,
        target_update,
    };
    let (grads, _report) = supervised_backward(&model, &batch).unwrap();

    for param in [
        GradientParam::W1(0),
        GradientParam::W1(config.perception_dims() + 1),
        GradientParam::B1(1),
        GradientParam::W2(0),
        GradientParam::W2(config.hidden_dims + 1),
        GradientParam::B2(0),
    ] {
        let actual = analytic_gradient(&grads, param);
        let expected = finite_difference_gradient(&model, &batch, param);
        assert!(
            (actual - expected).abs() < 2.0e-3,
            "{param:?}: analytic {actual}, finite difference {expected}"
        );
    }
}

#[test]
fn manifest_roundtrips_seeded_model() {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let model = NpaModel::seeded(config, 99);
    let manifest = BpkModelManifest::from_model(&model, grid, Some("unit-test".to_string()));
    let path = temp_path("manifest_roundtrip.json");

    burn_automata::import::save_manifest(&path, &manifest).unwrap();
    let loaded = burn_automata::import::load_manifest(&path).unwrap();
    fs::remove_file(&path).ok();

    assert_eq!(loaded.format_version, 1);
    assert_eq!(loaded.model_kind, "npa");
    assert_eq!(loaded.source.as_deref(), Some("unit-test"));
    assert_eq!(loaded.config, manifest.config);
    loaded.into_model().validate().unwrap();
}

#[test]
fn bpk_manifest_roundtrips_and_rejects_corruption() {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
    let model = NpaModel::seeded(config, 123);
    let manifest = BpkModelManifest::from_model(&model, grid, Some("binary-test".to_string()));
    let path = temp_path("manifest_roundtrip.bpk");

    let digest = burn_automata::import::save_manifest(&path, &manifest)
        .unwrap()
        .expect("bpk saves return payload digest");
    assert_eq!(digest.len(), 64);

    let loaded = burn_automata::import::load_manifest(&path).unwrap();
    assert_eq!(loaded.config, manifest.config);
    assert_eq!(loaded.weights.w1, manifest.weights.w1);

    let mut bytes = fs::read(&path).unwrap();
    fs::remove_file(&path).ok();
    let last = bytes.last_mut().unwrap();
    *last = last.wrapping_add(1);
    let err = burn_automata::import::decode_bpk_manifest(&bytes).unwrap_err();
    assert!(err.to_string().contains("checksum"));
}

#[test]
fn pytorch_npa_checkpoint_imports_to_bpk() {
    let input = temp_path("fake_npa.pth");
    let output = temp_path("fake_npa.bpk");
    write_fake_pytorch_checkpoint(&input);

    let report = burn_automata::import::import_model(&input, &output).unwrap();
    let loaded = burn_automata::import::load_manifest(&output).unwrap();
    fs::remove_file(&input).ok();
    fs::remove_file(&output).ok();

    assert_eq!(report.container, "bpk");
    assert!(report.sha256.is_some());
    assert_eq!(loaded.config.spatial_dims, 2);
    assert_eq!(loaded.config.state_dims, 1);
    assert_eq!(loaded.config.hidden_dims, 2);
    assert_eq!(loaded.hashgrid.eps, 0.1);
    assert_eq!(burn_automata::import::parameter_count(&loaded), 23);
}

#[test]
fn burn_tensor_bridge_preserves_shapes() {
    type B = burn::backend::NdArray;
    let device = Default::default();

    let positions = [[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]];
    let tensor = burn_automata::burn_bridge::positions_to_tensor::<B>(&positions, &device);
    assert_eq!(tensor.shape().dims(), [2, 4]);

    let states = [0.0, 1.0, 2.0, 3.0];
    let tensor = burn_automata::burn_bridge::states_to_tensor::<B>(&states, 2, 2, &device);
    assert_eq!(tensor.shape().dims(), [2, 2]);
}

#[test]
fn burn_tensor_forward_matches_cpu_update_rows() {
    type B = burn::backend::NdArray;

    let config = NpaConfig {
        hidden_dims: 8,
        ..NpaConfig::growing_2d()
    };
    let model = NpaModel::seeded(config.clone(), 5);
    let rows = 3;
    let features = (0..rows * config.perception_dims())
        .map(|idx| (idx as f32 * 0.03).sin())
        .collect::<Vec<_>>();
    let expected = model.forward_update_from_features(&features).unwrap();

    let device = Default::default();
    let tensor = burn::tensor::Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(features, [rows, config.perception_dims()]),
        &device,
    );
    let actual = model
        .forward_update_tensor::<B>(tensor, &device)
        .unwrap()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();

    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }
}

#[test]
fn scale_equivariant_cpu_step_preserves_scaled_rollout() {
    let (mut config, grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
    config.equivariance = EquivarianceMode::ParticleDensityAndScale;
    config.hidden_dims = 8;
    let model = NpaModel::seeded(config, 17);
    let particles = 32;
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        model.config.spatial_dims,
        23,
        ParticleSeed::UniformCircle,
        0.2,
    );

    let scale = 1.7;
    let mut scaled_grid = grid.clone();
    scaled_grid.eps *= scale;
    let scaled_positions = positions
        .iter()
        .map(|position| {
            [
                position[0] * scale,
                position[1] * scale,
                position[2] * scale,
                position[3],
            ]
        })
        .collect::<Vec<_>>();

    let base = model
        .step_cpu(&positions, &states, 1, particles, &grid, 1.0, None)
        .unwrap();
    let scaled = model
        .step_cpu(
            &scaled_positions,
            &states,
            1,
            particles,
            &scaled_grid,
            1.0,
            None,
        )
        .unwrap();

    for (base, scaled) in base.next_positions.iter().zip(scaled.next_positions.iter()) {
        for axis in 0..model.config.spatial_dims {
            let normalized = scaled[axis] / scale;
            assert!(
                (base[axis] - normalized).abs() < 2.0e-4,
                "position axis {axis}: base {}, scaled/scale {}",
                base[axis],
                normalized
            );
        }
    }
    for (base, scaled) in base.next_states.iter().zip(scaled.next_states.iter()) {
        assert!(
            (base - scaled).abs() < 2.0e-4,
            "state base {base}, scaled {scaled}"
        );
    }
}

#[test]
fn scale_equivariant_seed_hashgrid_scales_particle_eps_only() {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let scaled = config.hashgrid_for_seed_scale(&grid, 0.25, 0.72);
    assert!((scaled.eps - grid.eps * 0.25 / 0.72).abs() < 1.0e-7);
    assert_eq!(scaled.grid_size, grid.grid_size);
    assert_eq!(scaled.mode, grid.mode);

    let (texture_config, texture_grid) = NpaConfig::for_preset(AutomataPreset::Texture2d);
    let texture_scaled = texture_config.hashgrid_for_seed_scale(&texture_grid, 0.25, 1.0);
    assert_eq!(texture_scaled.eps, texture_grid.eps);

    let mut non_equivariant = config;
    non_equivariant.equivariance = EquivarianceMode::None;
    let unchanged = non_equivariant.hashgrid_for_seed_scale(&grid, 0.25, 0.72);
    assert_eq!(unchanged.eps, grid.eps);
}

#[test]
fn uv_torus_3d_seed_places_particles_on_colored_torus() {
    let particles = 256;
    let state_dims = 8;
    let scale = 0.72;
    let (positions, states) = seed_particles_scaled(
        2,
        particles,
        state_dims,
        3,
        29,
        ParticleSeed::UvTorus3d,
        scale,
    );
    let major = scale * UV_TORUS_INITIAL_SCALE;
    let minor = major * UV_TORUS_MINOR_RATIO;
    let mut max_target_error = 0.0_f32;
    let mut max_residual_error = 0.0_f32;
    let mut max_color_error = 0.0_f32;

    assert_eq!(positions.len(), particles * 2);
    assert_eq!(states.len(), particles * 2 * state_dims);
    for (idx, position) in positions.iter().enumerate() {
        let radial = (position[0] * position[0] + position[1] * position[1]).sqrt();
        let torus_radius = ((radial - major).powi(2) + position[2].powi(2)).sqrt();
        assert!(
            (torus_radius - minor).abs() < 2.0e-5,
            "particle {idx}: torus radius {torus_radius}, expected {minor}"
        );

        let state_base = idx * state_dims;
        let target = uv_torus_sample(idx % particles, particles, scale).position;
        let reconstructed = [
            position[0] + states[state_base],
            position[1] + states[state_base + 1],
            position[2] + states[state_base + 2],
        ];
        let target_error = ((reconstructed[0] - target[0]).powi(2)
            + (reconstructed[1] - target[1]).powi(2)
            + (reconstructed[2] - target[2]).powi(2))
        .sqrt();
        max_target_error = max_target_error.max(target_error);
        let residual_error = ((states[state_base] - (target[0] - position[0])).powi(2)
            + (states[state_base + 1] - (target[1] - position[1])).powi(2)
            + (states[state_base + 2] - (target[2] - position[2])).powi(2))
        .sqrt();
        max_residual_error = max_residual_error.max(residual_error);
        assert!((states[state_base + 3] - UV_TORUS_INITIAL_OPACITY_LOGIT).abs() < 1.0e-6);
        let actual_rgb = uv_torus_tail_state_to_rgb([
            states[state_base + state_dims - 3],
            states[state_base + state_dims - 2],
            states[state_base + state_dims - 1],
        ]);
        let expected_rgb = uv_torus_position_color(target, scale);
        let color_error = ((actual_rgb[0] - expected_rgb[0]).powi(2)
            + (actual_rgb[1] - expected_rgb[1]).powi(2)
            + (actual_rgb[2] - expected_rgb[2]).powi(2))
        .sqrt();
        max_color_error = max_color_error.max(color_error);
    }

    assert!(max_target_error <= 2.0e-5);
    assert!(max_residual_error <= 2.0e-5);
    assert!(max_color_error <= 1.0e-6);
}

#[test]
fn uv_torus_sampler_uses_independent_ring_and_tube_axes() {
    let particles = 256;
    let scale = 0.72;
    let first = uv_torus_sample(0, particles, scale);
    let same_tube_next_ring = uv_torus_sample(1, particles, scale);
    let next_tube_first_ring = uv_torus_sample(16, particles, scale);

    assert!(same_tube_next_ring.u > first.u);
    assert!((same_tube_next_ring.v - first.v).abs() <= f32::EPSILON);
    assert!((next_tube_first_ring.u - first.u).abs() <= f32::EPSILON);
    assert!(next_tube_first_ring.v > first.v);
}

#[test]
fn uv_torus_continuous_samples_use_implicit_surface_and_volume() {
    let scale = 0.72;
    let mut rng = StdRng::seed_from_u64(0x70_75);
    let mut max_surface_error = 0.0_f32;
    let mut max_projected_volume_error = 0.0_f32;
    let mut accumulated_surface_delta = 0.0_f32;
    let mut previous_surface = uv_torus_continuous_surface_position(&mut rng, scale);

    for _ in 0..128 {
        let surface = uv_torus_continuous_surface_position(&mut rng, scale);
        max_surface_error = max_surface_error.max(uv_torus_surface_error(surface, scale));
        accumulated_surface_delta += ((surface[0] - previous_surface[0]).powi(2)
            + (surface[1] - previous_surface[1]).powi(2)
            + (surface[2] - previous_surface[2]).powi(2))
        .sqrt();
        previous_surface = surface;

        let volume = uv_torus_continuous_volume_position(&mut rng, scale);
        let projected = uv_torus_project_position(volume, scale);
        max_projected_volume_error =
            max_projected_volume_error.max(uv_torus_surface_error(projected, scale));
    }

    assert!(max_surface_error <= 2.0e-5);
    assert!(max_projected_volume_error <= 2.0e-5);
    assert!(
        accumulated_surface_delta > scale,
        "continuous surface sampler did not cover a meaningful torus arc"
    );
}

#[test]
fn uv_torus_mesh_target_keeps_inner_and_outer_curvature_oriented() {
    let scale = 0.72;
    let minor = scale * UV_TORUS_MINOR_RATIO;
    let target = TriangleMeshTarget::torus(scale, minor, 96, 64).unwrap();

    let outer = target.project([scale + minor + 0.08, 0.0, 0.0]);
    let inner_hole = target.project([scale - minor - 0.08, 0.0, 0.0]);
    let inside_solid = target.project([scale + 0.5 * minor, 0.0, 0.0]);

    assert!(outer.signed_distance > 0.07);
    assert!(inner_hole.signed_distance > 0.07);
    assert!(inside_solid.signed_distance < -0.25);
    assert!(dot3(outer.normal, [1.0, 0.0, 0.0]) > 0.99);
    assert!(dot3(inner_hole.normal, [-1.0, 0.0, 0.0]) > 0.99);
    assert!(
        dot3(outer.normal, inner_hole.normal) < -0.99,
        "inner and outer tube normals should point in opposite directions"
    );
    assert!(uv_torus_surface_error(outer.closest, scale) < 2.0e-3);
    assert!(uv_torus_surface_error(inner_hole.closest, scale) < 2.0e-3);
    assert!((outer.signed_distance - uv_torus_signed_distance(outer.query, scale)).abs() < 2.0e-3);
    assert!(
        (inner_hole.signed_distance - uv_torus_signed_distance(inner_hole.query, scale)).abs()
            < 2.0e-3
    );
}

#[test]
fn mesh_surface_samples_cover_torus_surface_without_face_prefix_bias() {
    let scale = 0.72;
    let minor = scale * UV_TORUS_MINOR_RATIO;
    let target = TriangleMeshTarget::torus(scale, minor, 96, 64).unwrap();
    let ring_bins = 24usize;
    let tube_bins = 16usize;
    let mut covered_rings = HashSet::new();
    let mut covered_tubes = HashSet::new();

    for sample_idx in 0..512 {
        let sample = target.surface_sample(sample_idx);
        let theta = sample.position[1].atan2(sample.position[0]);
        let radial = (sample.position[0] * sample.position[0]
            + sample.position[1] * sample.position[1])
            .sqrt();
        let phi = sample.position[2].atan2(radial - scale);
        let ring = (((theta.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU)
            * ring_bins as f32)
            .floor() as usize)
            .min(ring_bins - 1);
        let tube = (((phi.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU)
            * tube_bins as f32)
            .floor() as usize)
            .min(tube_bins - 1);
        covered_rings.insert(ring);
        covered_tubes.insert(tube);
        assert!(
            uv_torus_surface_error(sample.position, scale) <= minor * 0.08,
            "sample {sample_idx} should remain near the torus surface"
        );
    }

    assert!(
        covered_rings.len() >= 20,
        "low sample counts should cover most torus rings, got {}",
        covered_rings.len()
    );
    assert!(
        covered_tubes.len() >= 12,
        "low sample counts should cover most torus tube bins, got {}",
        covered_tubes.len()
    );
}

#[test]
fn mesh_surface_samples_are_area_weighted() {
    let target = TriangleMeshTarget::new(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
        ],
        vec![[0, 1, 2], [0, 3, 4]],
    )
    .unwrap();
    let mut large_triangle_samples = 0usize;
    let samples = 256usize;
    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        if sample.position[0] + sample.position[1] > 1.15 {
            large_triangle_samples += 1;
        }
    }

    assert!(
        large_triangle_samples > samples * 3 / 4,
        "area-weighted sampling should strongly favor the larger triangle, got {large_triangle_samples}/{samples}"
    );
}

#[test]
fn mesh_random_surface_samples_are_area_weighted() {
    let target = TriangleMeshTarget::new(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
        ],
        vec![[0, 1, 2], [0, 3, 4]],
    )
    .unwrap();
    let mut rng = StdRng::seed_from_u64(0x5eed);
    let mut large_triangle_samples = 0usize;
    let samples = 512usize;
    for _ in 0..samples {
        let sample = target.random_surface_sample(&mut rng);
        if sample.position[0] + sample.position[1] > 1.15 {
            large_triangle_samples += 1;
        }
    }

    assert!(
        large_triangle_samples > samples * 3 / 4,
        "random surface sampling should favor area, got {large_triangle_samples}/{samples}"
    );
}

#[test]
fn utah_teapot_mesh_target_exposes_body_spout_handle_and_lid() {
    let scale = 0.72;
    let target = TriangleMeshTarget::utah_teapot(scale).unwrap();

    assert!(target.vertices.len() > 3_000);
    assert!(target.faces.len() > 8_000);
    assert_eq!(target.colors.as_ref().unwrap().len(), target.vertices.len());

    let (bounds_min, bounds_max) = target.bounds();
    assert!(bounds_min[0] < -0.75 * scale);
    assert!(bounds_max[0] > 0.65 * scale);
    assert!(bounds_min[1] < -0.45 * scale);
    assert!(bounds_max[1] > 0.45 * scale);
    assert!(bounds_min[2] < -0.45 * scale);
    assert!(bounds_max[2] > 0.45 * scale);

    let body = target.project([0.0, 0.0, -0.05 * scale]);
    let spout = target.project([0.82 * scale, 0.0, 0.05 * scale]);
    let handle = target.project([-0.92 * scale, 0.0, 0.02 * scale]);
    let lid = target.project([0.0, 0.0, 0.66 * scale]);

    assert!(body.closest[0].abs() < 0.58 * scale);
    assert!(spout.closest[0] > 0.55 * scale);
    assert!(handle.closest[0] < -0.70 * scale);
    assert!(lid.closest[2] > 0.38 * scale);
    for projection in [body, spout, handle, lid] {
        assert!(projection.distance.is_finite());
        assert!(projection.normal.iter().all(|value| value.is_finite()));
        assert!((dot3(projection.normal, projection.normal).sqrt() - 1.0).abs() < 5.0e-3);
        assert!(
            projection
                .color
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
        );
    }
}

#[test]
fn uv_torus_dense_3d_seed_uses_random_cloud_with_target_residuals() {
    let particles = 512;
    let state_dims = 8;
    let scale = 0.72;
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        state_dims,
        3,
        31,
        ParticleSeed::UvTorusDense3d,
        scale,
    );
    let dense_radius = uv_torus_dense_seed_radius(scale);
    let mut max_target_error = 0.0_f32;
    let mut max_residual_error = 0.0_f32;
    let mut max_color_error = 0.0_f32;
    let mut max_position_radius = 0.0_f32;

    for (idx, position) in positions.iter().enumerate() {
        let position_radius =
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt();
        max_position_radius = max_position_radius.max(position_radius);

        let state_base = idx * state_dims;
        let target = uv_torus_sample(idx, particles, scale).position;
        let reconstructed = [
            position[0] + states[state_base],
            position[1] + states[state_base + 1],
            position[2] + states[state_base + 2],
        ];
        let target_error = ((reconstructed[0] - target[0]).powi(2)
            + (reconstructed[1] - target[1]).powi(2)
            + (reconstructed[2] - target[2]).powi(2))
        .sqrt();
        max_target_error = max_target_error.max(target_error);
        let residual_error = ((states[state_base] - (target[0] - position[0])).powi(2)
            + (states[state_base + 1] - (target[1] - position[1])).powi(2)
            + (states[state_base + 2] - (target[2] - position[2])).powi(2))
        .sqrt();
        max_residual_error = max_residual_error.max(residual_error);
        assert!((states[state_base + 3] - UV_TORUS_INITIAL_OPACITY_LOGIT).abs() < 1.0e-6);
        let actual_rgb = uv_torus_tail_state_to_rgb([
            states[state_base + state_dims - 3],
            states[state_base + state_dims - 2],
            states[state_base + state_dims - 1],
        ]);
        let expected_rgb = uv_torus_position_color(target, scale);
        let color_error = ((actual_rgb[0] - expected_rgb[0]).powi(2)
            + (actual_rgb[1] - expected_rgb[1]).powi(2)
            + (actual_rgb[2] - expected_rgb[2]).powi(2))
        .sqrt();
        max_color_error = max_color_error.max(color_error);
    }

    assert!(max_position_radius <= dense_radius + 1.0e-6);
    assert!(max_target_error <= 2.0e-5);
    assert!(max_residual_error <= 2.0e-5);
    assert!(max_color_error <= 1.0e-6);
}

#[test]
fn teapot_morphogen_dense_3d_seed_uses_mesh_projected_seed_frame() {
    let particles = 256;
    let state_dims = 16;
    let scale = 0.72;
    let target = TriangleMeshTarget::utah_teapot(scale).unwrap();
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        state_dims,
        3,
        73,
        ParticleSeed::TeapotMorphogenDense3d,
        scale,
    );
    let mut max_projected_error = 0.0_f32;
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;

    for (idx, position) in positions.iter().enumerate() {
        min_x = min_x.min(position[0]);
        max_x = max_x.max(position[0]);
        let state_base = idx * state_dims;
        let projection = target.project([position[0], position[1], position[2]]);
        let projected = projection.closest;
        let reconstructed = [
            position[0] + states[state_base],
            position[1] + states[state_base + 1],
            position[2] + states[state_base + 2],
        ];
        let projected_error = ((reconstructed[0] - projected[0]).powi(2)
            + (reconstructed[1] - projected[1]).powi(2)
            + (reconstructed[2] - projected[2]).powi(2))
        .sqrt();
        max_projected_error = max_projected_error.max(projected_error);

        let actual_normal = [
            states[state_base + UV_TORUS_NORMAL_STATE_OFFSET],
            states[state_base + UV_TORUS_NORMAL_STATE_OFFSET + 1],
            states[state_base + UV_TORUS_NORMAL_STATE_OFFSET + 2],
        ];
        assert!((dot3(actual_normal, actual_normal).sqrt() - 1.0).abs() < 1.0e-5);
        assert!((states[state_base + 3] - UV_TORUS_INITIAL_OPACITY_LOGIT).abs() < 1.0e-6);

        let actual_rgb = uv_torus_tail_state_to_rgb([
            states[state_base + state_dims - 3],
            states[state_base + state_dims - 2],
            states[state_base + state_dims - 1],
        ]);
        let expected_rgb = projection.color;
        let color_error = ((actual_rgb[0] - expected_rgb[0]).powi(2)
            + (actual_rgb[1] - expected_rgb[1]).powi(2)
            + (actual_rgb[2] - expected_rgb[2]).powi(2))
        .sqrt();
        assert!(color_error <= 1.0e-6);
    }

    assert!(max_projected_error <= 2.0e-5);
    assert!(
        max_x - min_x > 1.2 * scale,
        "teapot dense seed should cover body, spout, and handle envelope"
    );
}

#[test]
fn teapot_field_dense_3d_seed_is_neutral_not_projected_or_precolored() {
    let particles = 256;
    let state_dims = 16;
    let scale = 0.72;
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        state_dims,
        3,
        73,
        ParticleSeed::TeapotFieldDense3d,
        scale,
    );
    let mut max_radius = 0.0_f32;

    for (idx, position) in positions.iter().enumerate() {
        let radius =
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt();
        max_radius = max_radius.max(radius);

        let state_base = idx * state_dims;
        assert_eq!(states[state_base], 0.0);
        assert_eq!(states[state_base + 1], 0.0);
        assert_eq!(states[state_base + 2], 0.0);
        assert!((states[state_base + 3] - UV_TORUS_INITIAL_OPACITY_LOGIT).abs() < 1.0e-6);
        assert_eq!(states[state_base + UV_TORUS_NORMAL_STATE_OFFSET], 0.0);
        assert_eq!(states[state_base + UV_TORUS_NORMAL_STATE_OFFSET + 1], 0.0);
        assert_eq!(states[state_base + UV_TORUS_NORMAL_STATE_OFFSET + 2], 0.0);
        assert_eq!(
            states[state_base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET],
            0.0
        );
        assert_eq!(states[state_base + state_dims - 3], 0.0);
        assert_eq!(states[state_base + state_dims - 2], 0.0);
        assert_eq!(states[state_base + state_dims - 1], 0.0);
    }

    assert!(max_radius <= scale + 1.0e-6);
}

#[test]
fn growth_3d_seeds_are_compact_neutral_and_not_target_assigned() {
    let particles = 512;
    let state_dims = 16;
    let scale = 0.72;
    for seed_mode in [ParticleSeed::TorusGrowth3d, ParticleSeed::TeapotGrowth3d] {
        let (positions, states) =
            seed_particles_scaled(1, particles, state_dims, 3, 73, seed_mode, scale);
        let mut max_radius = 0.0_f32;
        let mut min_radius = f32::MAX;
        let mut active_count = 0usize;
        let mut inactive_count = 0usize;

        for (idx, position) in positions.iter().enumerate() {
            let radius =
                (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                    .sqrt();
            max_radius = max_radius.max(radius);
            min_radius = min_radius.min(radius);

            let state_base = idx * state_dims;
            let opacity = states[state_base + 3];
            let material_opacity_channel = growth_3d_material_opacity_channel(state_dims);
            let domain_radius = growth_3d_domain_radius(scale).max(1.0e-4);
            for channel in 0..state_dims {
                if channel < 3 {
                    assert!(
                        (states[state_base + channel] - position[channel] / domain_radius).abs()
                            < 1.0e-6,
                        "{seed_mode:?} channel {channel} should store normalized seed-frame coordinate, not a target assignment"
                    );
                    continue;
                }
                if channel == 3 || Some(channel) == material_opacity_channel {
                    continue;
                }
                assert_eq!(
                    states[state_base + channel],
                    0.0,
                    "{seed_mode:?} should not seed particle identity or target channel {channel}"
                );
            }
            if radius <= growth_3d_active_core_radius(scale) {
                active_count += 1;
                assert!(
                    (opacity - GROWTH_3D_ACTIVE_OPACITY_LOGIT).abs() < 1.0e-6,
                    "{seed_mode:?} active opacity {opacity}"
                );
            } else {
                inactive_count += 1;
                assert!(
                    (opacity - GROWTH_3D_INACTIVE_OPACITY_LOGIT).abs() < 1.0e-6,
                    "{seed_mode:?} inactive opacity {opacity}"
                );
            }
        }

        assert!(max_radius <= growth_3d_seed_radius(scale) + 1.0e-6);
        assert!(
            min_radius < growth_3d_seed_radius(scale) * 0.35,
            "growth seed should include the compact core, got min radius {min_radius}"
        );
        assert!(
            max_radius > growth_3d_seed_radius(scale) * 0.85,
            "growth seed should fill most of the compact ball, got max radius {max_radius}"
        );
        assert!(
            active_count > 0,
            "{seed_mode:?} has no active core particles"
        );
        assert!(
            inactive_count > active_count * 4,
            "{seed_mode:?} should start from a sparse active core, active={active_count} inactive={inactive_count}"
        );
    }
}

#[test]
fn morphogen_seed_envelope_sampler_mixes_generic_callbacks() {
    let envelope = MorphogenSeedEnvelope {
        core_radius: 0.25,
        bounds_min: [3.0, -1.0, -1.0],
        bounds_max: [4.0, 1.0, 1.0],
        near_surface_jitter: 0.05,
    };
    let mut rng = StdRng::seed_from_u64(0x51eed);
    let mut saw_core = false;
    let mut saw_volume = false;
    let mut saw_surface = false;
    let mut saw_bounds = false;

    for _ in 0..512 {
        let position = morphogen_seed_envelope_position(
            &mut rng,
            envelope,
            |_| [0.0, 0.0, 0.0],
            |_| [1.0, 0.0, 0.0],
            |_| [2.0, 0.0, 0.0],
            |_| [1.0, 0.0, 0.0],
        );
        saw_core |= position[0].abs() <= 1.0e-6;
        saw_volume |= (position[0] - 1.0).abs() <= 1.0e-6;
        saw_surface |= (1.95..=2.05).contains(&position[0]);
        saw_bounds |= (3.0..=4.0).contains(&position[0]);
    }

    assert!(saw_core, "generic envelope never sampled core callback");
    assert!(saw_volume, "generic envelope never sampled volume callback");
    assert!(
        saw_surface,
        "generic envelope never sampled near-surface callback"
    );
    assert!(saw_bounds, "generic envelope never sampled bounds callback");
}

#[test]
fn normalized_seed_scale_preserves_hashgrid_occupancy_for_scaled_3d_seeds() {
    let (config, base_grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let particles = 8192;
    let state_dims = config.state_dims;
    let reference_scale = 0.72;
    let baseline = seed_occupancy_stats(
        &base_grid,
        particles,
        state_dims,
        reference_scale,
        reference_scale,
        false,
    );

    for scale in [0.04_f32, 0.16, 1.2] {
        let grid = config.hashgrid_for_seed_scale(&base_grid, scale, reference_scale);
        let stats =
            seed_occupancy_stats(&grid, particles, state_dims, scale, reference_scale, false);
        assert_eq!(
            stats, baseline,
            "scale-normalized hashgrid should preserve occupancy at scale {scale}"
        );
    }

    let unnormalized_small = seed_occupancy_stats(
        &base_grid,
        particles,
        state_dims,
        0.04,
        reference_scale,
        false,
    );
    assert!(
        unnormalized_small.1 > baseline.1 * 40,
        "fixed eps should expose the dense-cell failure mode: baseline={baseline:?} fixed={unnormalized_small:?}"
    );
    assert!(
        unnormalized_small.0 < baseline.0 / 100,
        "fixed eps should collapse the small seed into far fewer cells: baseline={baseline:?} fixed={unnormalized_small:?}"
    );
}

fn seed_occupancy_stats(
    grid: &burn_automata::kernels::HashGridConfig,
    particles: usize,
    state_dims: usize,
    seed_scale: f32,
    _reference_scale: f32,
    _normalize: bool,
) -> (usize, usize) {
    let (positions, _states) = seed_particles_scaled(
        1,
        particles,
        state_dims,
        3,
        42,
        ParticleSeed::TorusMorphogenDense3d,
        seed_scale,
    );
    let snapshot = build_hashgrid(&positions, 1, particles, grid).unwrap();
    snapshot
        .bin_offsets
        .windows(2)
        .map(|window| window[1] - window[0])
        .fold((0usize, 0usize), |(nonempty, max), count| {
            (nonempty + usize::from(count > 0), max.max(count))
        })
}

#[test]
fn torus_field_dense_3d_seed_is_neutral_not_index_assigned() {
    let particles = 128;
    let state_dims = 8;
    let scale = 0.72;
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        state_dims,
        3,
        43,
        ParticleSeed::TorusFieldDense3d,
        scale,
    );
    let dense_radius = uv_torus_dense_seed_radius(scale);

    for (idx, position) in positions.iter().enumerate() {
        let radius =
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt();
        assert!(radius <= dense_radius + 1.0e-6);

        let state_base = idx * state_dims;
        assert_eq!(states[state_base], 0.0);
        assert_eq!(states[state_base + 1], 0.0);
        assert_eq!(states[state_base + 2], 0.0);
        assert!((states[state_base + 3] - UV_TORUS_INITIAL_OPACITY_LOGIT).abs() < 1.0e-6);
        assert_eq!(states[state_base + state_dims - 3], 0.0);
        assert_eq!(states[state_base + state_dims - 2], 0.0);
        assert_eq!(states[state_base + state_dims - 1], 0.0);
    }
}

#[test]
fn torus_morphogen_dense_3d_seed_uses_projected_seed_frame_not_index() {
    let particles = 128;
    let state_dims = 8;
    let scale = 0.72;
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        state_dims,
        3,
        43,
        ParticleSeed::TorusMorphogenDense3d,
        scale,
    );
    let mut max_projected_error = 0.0_f32;
    let mut max_index_error = 0.0_f32;
    let mut min_radial = f32::MAX;
    let mut max_radial = f32::MIN;

    for (idx, position) in positions.iter().enumerate() {
        let radius =
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt();
        assert!(radius <= uv_torus_outer_radius(scale) * 1.9);
        let radial = (position[0] * position[0] + position[1] * position[1]).sqrt();
        min_radial = min_radial.min(radial);
        max_radial = max_radial.max(radial);

        let state_base = idx * state_dims;
        let projected = uv_torus_project_position([position[0], position[1], position[2]], scale);
        let reconstructed = [
            position[0] + states[state_base],
            position[1] + states[state_base + 1],
            position[2] + states[state_base + 2],
        ];
        let projected_error = ((reconstructed[0] - projected[0]).powi(2)
            + (reconstructed[1] - projected[1]).powi(2)
            + (reconstructed[2] - projected[2]).powi(2))
        .sqrt();
        max_projected_error = max_projected_error.max(projected_error);

        let indexed = uv_torus_sample(idx, particles, scale).position;
        let index_error = ((reconstructed[0] - indexed[0]).powi(2)
            + (reconstructed[1] - indexed[1]).powi(2)
            + (reconstructed[2] - indexed[2]).powi(2))
        .sqrt();
        max_index_error = max_index_error.max(index_error);

        assert!((states[state_base + 3] - UV_TORUS_INITIAL_OPACITY_LOGIT).abs() < 1.0e-6);
        let actual_rgb = uv_torus_tail_state_to_rgb([
            states[state_base + state_dims - 3],
            states[state_base + state_dims - 2],
            states[state_base + state_dims - 1],
        ]);
        let expected_rgb = uv_torus_position_color(projected, scale);
        let color_error = ((actual_rgb[0] - expected_rgb[0]).powi(2)
            + (actual_rgb[1] - expected_rgb[1]).powi(2)
            + (actual_rgb[2] - expected_rgb[2]).powi(2))
        .sqrt();
        assert!(color_error <= 1.0e-6);
    }

    assert!(max_projected_error <= 2.0e-5);
    assert!(
        min_radial < scale * 0.35 && max_radial > uv_torus_outer_radius(scale) * 0.75,
        "morphogen seed should cover both core and torus target envelope"
    );
    assert!(
        max_index_error >= 0.1,
        "morphogen seed unexpectedly matched indexed target error {max_index_error}"
    );
}

#[test]
fn torus_morphogen_seed_initializes_orientation_channels_when_available() {
    let particles = 64;
    let state_dims = 16;
    let scale = 0.72;
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        state_dims,
        3,
        47,
        ParticleSeed::TorusMorphogenDense3d,
        scale,
    );
    assert!(uv_torus_orientation_state_available(state_dims));

    for (idx, position) in positions.iter().enumerate() {
        let state_base = idx * state_dims;
        let source = [position[0], position[1], position[2]];
        let actual_normal = [
            states[state_base + UV_TORUS_NORMAL_STATE_OFFSET],
            states[state_base + UV_TORUS_NORMAL_STATE_OFFSET + 1],
            states[state_base + UV_TORUS_NORMAL_STATE_OFFSET + 2],
        ];
        let expected_normal = uv_torus_outward_normal(source, scale);
        let normal_len = dot3(actual_normal, actual_normal).sqrt();
        assert!((normal_len - 1.0).abs() < 1.0e-5);
        assert!(dot3(actual_normal, expected_normal) > 0.999);
        assert!(
            (states[state_base + UV_TORUS_SIGNED_DISTANCE_STATE_OFFSET]
                - uv_torus_signed_distance(source, scale))
            .abs()
                < 1.0e-6
        );

        let projected = uv_torus_project_position(source, scale);
        let actual_rgb = uv_torus_tail_state_to_rgb([
            states[state_base + state_dims - 3],
            states[state_base + state_dims - 2],
            states[state_base + state_dims - 1],
        ]);
        let expected_rgb = uv_torus_position_color(projected, scale);
        let color_error = ((actual_rgb[0] - expected_rgb[0]).powi(2)
            + (actual_rgb[1] - expected_rgb[1]).powi(2)
            + (actual_rgb[2] - expected_rgb[2]).powi(2))
        .sqrt();
        assert!(color_error <= 1.0e-6);
    }
}

#[test]
fn uv_torus_zero_update_artifact_roundtrips_and_preserves_seed() {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let model = NpaModel {
        config: config.clone(),
        weights: NpaWeights::zeros(&config),
    };
    let manifest = BpkModelManifest::from_model(
        &model,
        grid.clone(),
        Some("unit-test:uv-torus-3d".to_string()),
    );
    let path = temp_path("uv_torus_3d.bpk");

    burn_automata::import::save_manifest(&path, &manifest).unwrap();
    let loaded = burn_automata::import::load_manifest(&path).unwrap();
    fs::remove_file(&path).ok();
    let loaded_model = loaded.into_model();

    let cfg = RolloutConfig {
        particle_count: 128,
        steps: 4,
        update_prob: 1.0,
        seed_scale: 0.72,
        ..RolloutConfig::default()
    };
    let trace = run_rollout(&loaded_model, &grid, &cfg, ParticleSeed::UvTorus3d).unwrap();
    let (seed_positions, seed_states) = seed_particles_scaled(
        1,
        cfg.particle_count,
        loaded_model.config.state_dims,
        loaded_model.config.spatial_dims,
        cfg.seed,
        ParticleSeed::UvTorus3d,
        cfg.seed_scale,
    );

    assert!(trace.mean_dx.iter().all(|value| value.abs() < 1.0e-8));
    assert_eq!(trace.positions, seed_positions);
    assert_eq!(trace.states, seed_states);
}

#[test]
fn uv_torus_opacity_growth_model_increases_visibility_state() {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let mut weights = NpaWeights::zeros(&config);
    weights.b2[config.spatial_dims + 3] = UV_TORUS_OPACITY_GROWTH_DELTA;
    let model = NpaModel {
        config: config.clone(),
        weights,
    };
    let cfg = RolloutConfig {
        particle_count: 128,
        steps: 10,
        update_prob: 1.0,
        seed_scale: 0.72,
        ..RolloutConfig::default()
    };
    let trace = run_rollout(&model, &grid, &cfg, ParticleSeed::UvTorus3d).unwrap();
    let expected =
        UV_TORUS_INITIAL_OPACITY_LOGIT + UV_TORUS_OPACITY_GROWTH_DELTA * cfg.steps as f32 * cfg.dt;

    for state in trace.states.chunks_exact(config.state_dims) {
        assert!(
            (state[3] - expected).abs() < 1.0e-5,
            "opacity state {}, expected {expected}",
            state[3]
        );
    }
    assert!(trace.mean_dx.iter().all(|value| value.abs() < 1.0e-8));
}

#[test]
fn shipped_3d_growth_assets_are_local_dynamic_and_not_target_seeded() {
    for (relative_path, seed_mode, expected_source) in [
        (
            "assets/models/uv_torus_growth_3d.bpk",
            ParticleSeed::TorusGrowth3d,
            "render-refined-rust:ablation-rust:uv-torus-3d:conditionless-local-random-ball-rollout-ablation",
        ),
        (
            "assets/models/teapot_growth_3d.bpk",
            ParticleSeed::TeapotGrowth3d,
            "retimed-local-front:hidden=skipped:gain=2:alpha=1:front_retime=false:active_opacity_hidden=skipped:active_opacity_gain=skipped:opacity_bias=skipped:material_opacity_bias=0.55:base=render-refined-rust:ablation-rust:utah-teapot-2026:conditionless-local-random-ball-rollout-ablation",
        ),
    ] {
        let path = workspace_path(relative_path);
        let manifest = burn_automata::import::load_manifest(path).unwrap();
        assert_eq!(manifest.model_kind, "npa", "{relative_path}");
        assert_eq!(manifest.config.spatial_dims, 3, "{relative_path}");
        assert!(
            !manifest.config.position_features,
            "{relative_path} must not use absolute world-position features"
        );
        let source = manifest.source.as_deref().unwrap_or_default();
        assert_eq!(
            source, expected_source,
            "{relative_path} should stay on the reviewed latest dynamic 3D growth artifact"
        );
        assert!(
            (source.starts_with("render-refined-rust:")
                || source.starts_with("retimed-local-front:"))
                && source.contains("conditionless-local")
                && !source.contains("position-field")
                && !source.contains("seed-frame")
                && !source.contains("render-proxy-rust"),
            "{relative_path} must use latest local render-refinement lineage without target-assigned shortcuts, source={source}"
        );
        let grid = manifest.hashgrid.clone();
        let model = manifest.into_model();
        let cfg = RolloutConfig {
            particle_count: 512,
            steps: 64,
            update_prob: 1.0,
            seed: CATALOG_3D_GROWTH_SEED,
            seed_scale: 0.72,
            ..RolloutConfig::default()
        };
        let (_seed_positions, seed_states) = seed_particles_scaled(
            1,
            cfg.particle_count,
            model.config.state_dims,
            model.config.spatial_dims,
            cfg.seed,
            seed_mode,
            cfg.seed_scale,
        );
        let active_seed_count = seed_states
            .chunks_exact(model.config.state_dims)
            .filter(|state| state[3] > -1.0)
            .count();
        assert!(
            active_seed_count > 0 && active_seed_count < cfg.particle_count / 8,
            "{relative_path} should start from a sparse active core, active={active_seed_count}"
        );

        let trace = run_rollout(&model, &grid, &cfg, seed_mode).unwrap();
        let initial_color_state = color_state_stats(&seed_states, model.config.state_dims);
        let final_color_state = color_state_stats(&trace.states, model.config.state_dims);
        assert!(
            initial_color_state.active_max_abs <= 1.0e-6,
            "{relative_path} should not precolor the sparse seed core: {initial_color_state:?}"
        );
        assert!(
            final_color_state.active_mean_abs >= initial_color_state.active_mean_abs + 0.02
                && final_color_state.active_max_abs >= 0.05,
            "{relative_path} should grow visible color state from neutral seed: initial={initial_color_state:?} final={final_color_state:?}"
        );
        assert!(
            final_color_state.active_channel_stddev_mean >= 0.02,
            "{relative_path} final color state should vary across active particles instead of becoming a uniform tint: {final_color_state:?}"
        );
        let max_motion = trace.mean_dx.iter().copied().fold(0.0_f32, f32::max);
        let mut max_radius = 0.0_f32;
        let mut max_abs_z = 0.0_f32;
        let mut min_opacity = f32::MAX;
        let mut max_opacity = f32::MIN;
        for (idx, position) in trace.positions.iter().enumerate() {
            assert!(position.iter().all(|value| value.is_finite()));
            max_radius = max_radius.max(
                (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                    .sqrt(),
            );
            max_abs_z = max_abs_z.max(position[2].abs());
            let opacity = trace.states[idx * model.config.state_dims + 3];
            min_opacity = min_opacity.min(opacity);
            max_opacity = max_opacity.max(opacity);
        }
        assert!(trace.mean_dx.iter().all(|value| value.is_finite()));
        assert!(
            max_motion > 0.01,
            "{relative_path} should move from the sparse-core seed, max mean dx={max_motion}"
        );
        assert!(
            max_radius > growth_3d_seed_radius(cfg.seed_scale),
            "{relative_path} should expand beyond the compact seed, max radius={max_radius}"
        );
        assert!(
            max_abs_z > cfg.seed_scale * 0.25,
            "{relative_path} should use 3D volume, max |z|={max_abs_z}"
        );
        assert!(
            min_opacity.is_finite() && max_opacity.is_finite() && max_opacity < 24.0,
            "{relative_path} opacity state should stay finite and bounded, min={min_opacity} max={max_opacity}"
        );
    }
}

#[test]
fn shipped_3d_growth_assets_remain_bounded_across_seed_sweep() {
    for (relative_path, seed_mode) in [
        (
            "assets/models/uv_torus_growth_3d.bpk",
            ParticleSeed::TorusGrowth3d,
        ),
        (
            "assets/models/teapot_growth_3d.bpk",
            ParticleSeed::TeapotGrowth3d,
        ),
    ] {
        let manifest = burn_automata::import::load_manifest(workspace_path(relative_path)).unwrap();
        let grid = manifest.hashgrid.clone();
        let model = manifest.into_model();
        for seed in [CATALOG_3D_GROWTH_SEED, 42, 99, 1234] {
            let cfg = RolloutConfig {
                particle_count: 512,
                steps: 64,
                update_prob: 1.0,
                seed,
                seed_scale: 0.72,
                ..RolloutConfig::default()
            };
            let (_seed_positions, seed_states) = seed_particles_scaled(
                1,
                cfg.particle_count,
                model.config.state_dims,
                model.config.spatial_dims,
                cfg.seed,
                seed_mode,
                cfg.seed_scale,
            );
            let active_seed_count = seed_states
                .chunks_exact(model.config.state_dims)
                .filter(|state| state[3] > -1.0)
                .count();
            let trace = run_rollout(&model, &grid, &cfg, seed_mode).unwrap();
            let max_motion = trace.mean_dx.iter().copied().fold(0.0_f32, f32::max);
            let mut final_active_count = 0usize;
            let mut max_opacity = f32::NEG_INFINITY;
            for state in trace.states.chunks_exact(model.config.state_dims) {
                let opacity = state[3];
                assert!(
                    opacity.is_finite(),
                    "{relative_path} seed {seed} produced non-finite opacity {opacity}"
                );
                max_opacity = max_opacity.max(opacity);
                if opacity > -1.0 {
                    final_active_count += 1;
                }
            }
            assert!(
                active_seed_count > 0 && active_seed_count < cfg.particle_count / 8,
                "{relative_path} seed {seed} should start from sparse growth core, active={active_seed_count}"
            );
            assert!(
                final_active_count > active_seed_count * 4,
                "{relative_path} seed {seed} should grow visible particles, active={active_seed_count}->{final_active_count}"
            );
            assert!(
                max_motion > 0.01,
                "{relative_path} seed {seed} should move dynamically, max mean dx={max_motion}"
            );
            assert!(
                max_opacity < 24.0,
                "{relative_path} seed {seed} opacity should stay bounded, max={max_opacity}"
            );
        }
    }
}

#[test]
fn shipped_3d_growth_assets_are_strictly_measured_before_promotion() {
    for case in [
        GrowthValidationCase::torus("assets/models/uv_torus_growth_3d.bpk"),
        GrowthValidationCase::teapot("assets/models/teapot_growth_3d.bpk"),
    ] {
        let report = strict_growth_validation_report(case);

        assert!(
            !report.position_features,
            "{} must stay local and must not use absolute world-position features",
            case.relative_path
        );
        assert!(
            report.non_opacity_seed_abs_max <= 1.0e-6,
            "{} seeds target or identity state outside opacity: max abs {}",
            case.relative_path,
            report.non_opacity_seed_abs_max
        );
        assert!(
            report.active_seed_count > 0 && report.active_seed_count < case.particle_count / 8,
            "{} should initialize like 2D growth with a sparse active core, active={}",
            case.relative_path,
            report.active_seed_count
        );
        assert!(
            report.final_active_count > report.active_seed_count * 4,
            "{} should activate substantially more particles than the seed core: seed_active={} final_active={}",
            case.relative_path,
            report.active_seed_count,
            report.final_active_count
        );
        assert!(
            report.newly_activated_fraction >= 0.50,
            "{} should activate at least half of initially inactive particles, activated={} fraction={}",
            case.relative_path,
            report.newly_activated_count,
            report.newly_activated_fraction
        );
        assert!(
            report.final_active_max_radius > growth_3d_seed_radius(case.seed_scale),
            "{} active front should expand beyond the initial seed ball, active_mean_radius={} active_max_radius={}",
            case.relative_path,
            report.final_active_mean_radius,
            report.final_active_max_radius
        );
        assert!(
            report.max_motion_per_step > 0.01,
            "{} appears static from the compact seed: max mean dx={}",
            case.relative_path,
            report.max_motion_per_step
        );
        assert!(
            report.mean_final_displacement > growth_3d_seed_radius(case.seed_scale),
            "{} should actually grow out of the compact seed, mean displacement={}",
            case.relative_path,
            report.mean_final_displacement
        );
        assert!(
            !report.strict_passed,
            "{} unexpectedly passed the strict local-3D morphogenesis gate; promote it by replacing this guard with the positive gate",
            case.relative_path
        );
        if matches!(case.target, GrowthTarget::Teapot) {
            assert!(
                !report.temporal_progressive_activation,
                "{} should remain blocked on seed-varied temporal activation until robust retraining replaces the shipped artifact: {report:?}",
                case.relative_path
            );
        }
        assert!(
            report.temporal_geometry_progressive.is_finite(),
            "{} temporal geometry progress should be measured: {report:?}",
            case.relative_path
        );
        if matches!(case.target, GrowthTarget::Teapot) {
            assert!(
                report.temporal_geometry_progressive.passed,
                "{} teapot geometry should remain progressive under corrected target sampling: {report:?}",
                case.relative_path
            );
        } else {
            assert!(
                !report.temporal_geometry_progressive.passed,
                "{} torus should expose the corrected full-surface coverage blocker until retrained: {report:?}",
                case.relative_path
            );
        }
        assert!(
            report.front_coherence.passed,
            "{} should activate through a local front instead of waking distant target particles directly: {report:?}",
            case.relative_path
        );
        assert!(
            report.front_coherence.transition_count >= 2
                && report.front_coherence.newly_activated_count > 0,
            "{} should grow through multiple measured local-front transitions: {report:?}",
            case.relative_path
        );
        assert!(
            report.front_coherence.local_newly_activated_fraction >= 0.90
                && report.front_coherence.max_nearest_previous_active_distance
                    <= report.front_coherence.max_allowed_distance * 1.05,
            "{} front coherence distances should remain bounded: {report:?}",
            case.relative_path
        );
        assert!(
            report.front_coherence.mean_nearest_previous_active_distance
                <= report.front_coherence.max_allowed_distance * 0.75,
            "{} mean local-front activation distance should stay comfortably below the threshold: {report:?}",
            case.relative_path
        );
        if matches!(case.target, GrowthTarget::Torus) {
            assert!(
                report.final_target_coverage.covered_fraction < 0.60,
                "{} unexpectedly passed full-torus target coverage; replace this guard with a positive assertion after the next artifact promotion",
                case.relative_path
            );
        } else {
            assert!(
                report.final_target_coverage.covered_fraction >= 0.60,
                "{} teapot should now pass target coverage under corrected surface sampling: {report:?}",
                case.relative_path
            );
        }
        if matches!(case.target, GrowthTarget::Teapot) {
            assert!(
                report.render_density_psnr_db >= 10.0,
                "{} teapot diagnostic should retain render-density PSNR even while robust temporal activation is blocked, got {}",
                case.relative_path,
                report.render_density_psnr_db
            );
        } else {
            assert!(
                report.render_density_psnr_db < 10.0,
                "{} strict gate should fail specifically on shape density today, got density PSNR {}",
                case.relative_path,
                report.render_density_psnr_db
            );
        }
        assert!(
            report.final_surface.mean.is_finite()
                && report.final_surface.max.is_finite()
                && report.initial_surface.mean.is_finite()
                && report.initial_surface.max.is_finite()
                && report.final_active_surface.mean.is_finite()
                && report.final_active_surface.max.is_finite()
                && report.initial_active_surface.mean.is_finite()
                && report.initial_active_surface.max.is_finite(),
            "{} surface metrics should be finite: {report:?}",
            case.relative_path
        );
        assert!(
            report.initial_target_coverage.mean.is_finite()
                && report.initial_target_coverage.max.is_finite()
                && report.initial_target_coverage.covered_fraction.is_finite()
                && report.final_target_coverage.mean.is_finite()
                && report.final_target_coverage.max.is_finite()
                && report.final_target_coverage.covered_fraction.is_finite(),
            "{} target coverage metrics should be finite: {report:?}",
            case.relative_path
        );
        assert!(
            report.render_color_psnr_db.is_finite() && report.render_depth_psnr_db.is_finite(),
            "{} render color/depth metrics should be finite: {report:?}",
            case.relative_path
        );
    }
}

#[test]
fn shipped_3d_growth_assets_are_dynamic_but_render_gap_is_measured() {
    for case in [
        CatalogRenderSanityCase {
            validation: GrowthValidationCase::torus("assets/models/uv_torus_growth_3d.bpk"),
            max_total_loss: 1.60,
            min_density_psnr_db: -1.85,
            min_color_psnr_db: 12.0,
            min_depth_psnr_db: 18.5,
        },
        CatalogRenderSanityCase {
            validation: GrowthValidationCase::teapot("assets/models/teapot_growth_3d.bpk"),
            max_total_loss: 0.25,
            min_density_psnr_db: 7.5,
            min_color_psnr_db: 15.0,
            min_depth_psnr_db: 28.0,
        },
    ] {
        let report = catalog_render_sanity_report(case.validation);
        assert!(
            report.total_loss <= case.max_total_loss
                && report.density_psnr_db >= case.min_density_psnr_db
                && report.color_psnr_db >= case.min_color_psnr_db
                && report.depth_psnr_db >= case.min_depth_psnr_db,
            "{} regressed below the latest dynamic-artifact render floor: {report:?}",
            case.validation.relative_path
        );
        assert!(
            report.density_psnr_db < 10.0,
            "{} 512-particle catalog sanity should still record the current low-count render-density gap: {report:?}",
            case.validation.relative_path
        );
    }
}

#[test]
#[ignore = "acceptance gate for the next promoted local 3D artifacts"]
fn promoted_3d_growth_assets_pass_strict_morphogenesis_gate() {
    for case in [
        GrowthValidationCase::torus("assets/models/uv_torus_growth_3d.bpk"),
        GrowthValidationCase::teapot("assets/models/teapot_growth_3d.bpk"),
    ] {
        let report = strict_growth_validation_report(case);
        assert!(
            report.strict_passed,
            "{} did not pass strict morphogenesis gate: {report:?}",
            case.relative_path
        );
    }
}

#[derive(Clone, Copy, Debug)]
struct GrowthValidationCase {
    relative_path: &'static str,
    target: GrowthTarget,
    seed_mode: ParticleSeed,
    seed_scale: f32,
    particle_count: usize,
    steps: usize,
}

impl GrowthValidationCase {
    fn torus(relative_path: &'static str) -> Self {
        Self {
            relative_path,
            target: GrowthTarget::Torus,
            seed_mode: ParticleSeed::TorusGrowth3d,
            seed_scale: 0.72,
            particle_count: 512,
            steps: 64,
        }
    }

    fn teapot(relative_path: &'static str) -> Self {
        Self {
            relative_path,
            target: GrowthTarget::Teapot,
            seed_mode: ParticleSeed::TeapotGrowth3d,
            seed_scale: 0.72,
            particle_count: 1024,
            steps: 64,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum GrowthTarget {
    Torus,
    Teapot,
}

#[derive(Clone, Copy, Debug)]
struct CatalogRenderSanityCase {
    validation: GrowthValidationCase,
    max_total_loss: f32,
    min_density_psnr_db: f32,
    min_color_psnr_db: f32,
    min_depth_psnr_db: f32,
}

#[derive(Debug)]
struct StrictGrowthValidationReport {
    position_features: bool,
    active_seed_count: usize,
    final_active_count: usize,
    newly_activated_count: usize,
    newly_activated_fraction: f32,
    final_active_mean_radius: f32,
    final_active_max_radius: f32,
    non_opacity_seed_abs_max: f32,
    mean_final_displacement: f32,
    initial_surface: SurfaceStats,
    final_surface: SurfaceStats,
    initial_active_surface: SurfaceStats,
    final_active_surface: SurfaceStats,
    initial_target_coverage: TargetCoverageStats,
    final_target_coverage: TargetCoverageStats,
    max_motion_per_step: f32,
    render_density_psnr_db: f32,
    render_color_psnr_db: f32,
    render_depth_psnr_db: f32,
    temporal_progressive_activation: bool,
    temporal_geometry_progressive: MeasuredBool,
    front_coherence: FrontCoherenceReport,
    strict_passed: bool,
}

#[derive(Clone, Copy, Debug)]
struct ColorStateStats {
    active_mean_abs: f32,
    active_max_abs: f32,
    active_channel_stddev_mean: f32,
}

#[derive(Clone, Copy, Debug)]
struct MeasuredBool {
    passed: bool,
    surface_mean_ratio: f32,
    target_coverage_mean_ratio: f32,
    target_coverage_fraction_delta: f32,
}

impl MeasuredBool {
    fn is_finite(self) -> bool {
        self.surface_mean_ratio.is_finite()
            && self.target_coverage_mean_ratio.is_finite()
            && self.target_coverage_fraction_delta.is_finite()
    }
}

#[derive(Clone, Copy, Debug)]
struct FrontCoherenceReport {
    passed: bool,
    transition_count: usize,
    newly_activated_count: usize,
    local_newly_activated_fraction: f32,
    mean_nearest_previous_active_distance: f32,
    max_nearest_previous_active_distance: f32,
    max_allowed_distance: f32,
}

#[derive(Debug)]
struct CatalogRenderSanityReport {
    total_loss: f32,
    density_psnr_db: f32,
    color_psnr_db: f32,
    depth_psnr_db: f32,
}

fn catalog_render_sanity_report(case: GrowthValidationCase) -> CatalogRenderSanityReport {
    let manifest =
        burn_automata::import::load_manifest(workspace_path(case.relative_path)).unwrap();
    let grid = manifest.hashgrid.clone();
    let model = manifest.into_model();
    let cfg = RolloutConfig {
        particle_count: 512,
        steps: case.steps,
        update_prob: 1.0,
        seed: CATALOG_3D_GROWTH_SEED,
        seed_scale: case.seed_scale,
        ..RolloutConfig::default()
    };
    let trace = run_rollout(&model, &grid, &cfg, case.seed_mode).unwrap();
    let target = match case.target {
        GrowthTarget::Torus => TriangleMeshTarget::torus(
            case.seed_scale,
            case.seed_scale * UV_TORUS_MINOR_RATIO,
            64,
            48,
        )
        .unwrap(),
        GrowthTarget::Teapot => TriangleMeshTarget::utah_teapot(case.seed_scale).unwrap(),
    };
    let render = mesh_multiview_render_loss_from_trace(
        &trace,
        &target,
        RenderLossConfig {
            image_size: 48,
            target_samples: 1024,
            world_scale: case.seed_scale * 2.0,
            ..RenderLossConfig::default()
        },
    )
    .unwrap();

    CatalogRenderSanityReport {
        total_loss: render.total_loss,
        density_psnr_db: render.density_psnr_db,
        color_psnr_db: render.color_psnr_db,
        depth_psnr_db: render.depth_psnr_db,
    }
}

fn strict_growth_validation_report(case: GrowthValidationCase) -> StrictGrowthValidationReport {
    let manifest =
        burn_automata::import::load_manifest(workspace_path(case.relative_path)).unwrap();
    let position_features = manifest.config.position_features;
    let grid = manifest.hashgrid.clone();
    let model = manifest.into_model();
    let cfg = RolloutConfig {
        particle_count: case.particle_count,
        steps: case.steps,
        update_prob: 1.0,
        seed: CATALOG_3D_GROWTH_SEED,
        seed_scale: case.seed_scale,
        ..RolloutConfig::default()
    };
    let (seed_positions, seed_states) = seed_particles_scaled(
        1,
        cfg.particle_count,
        model.config.state_dims,
        model.config.spatial_dims,
        cfg.seed,
        case.seed_mode,
        cfg.seed_scale,
    );
    let mut active_seed_count = 0usize;
    let mut non_opacity_seed_abs_max = 0.0_f32;
    let mut seed_active = Vec::with_capacity(cfg.particle_count);
    let material_opacity_channel = growth_3d_material_opacity_channel(model.config.state_dims);
    for state in seed_states.chunks_exact(model.config.state_dims) {
        let active = state[3] > -1.0;
        seed_active.push(active);
        if active {
            active_seed_count += 1;
        }
        for (channel, value) in state.iter().enumerate() {
            if channel != 3 && Some(channel) != material_opacity_channel && channel >= 3 {
                non_opacity_seed_abs_max = non_opacity_seed_abs_max.max(value.abs());
            }
        }
    }

    let trace = run_rollout(&model, &grid, &cfg, case.seed_mode).unwrap();
    let mut final_active_count = 0usize;
    let mut newly_activated_count = 0usize;
    let mut final_active_radius_sum = 0.0_f32;
    let mut final_active_max_radius = 0.0_f32;
    for (idx, position) in trace.positions.iter().enumerate() {
        let opacity = trace.states[idx * model.config.state_dims + 3];
        if opacity > -1.0 {
            final_active_count += 1;
            if !seed_active[idx] {
                newly_activated_count += 1;
            }
            let radius =
                (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                    .sqrt();
            final_active_radius_sum += radius;
            final_active_max_radius = final_active_max_radius.max(radius);
        }
    }
    let inactive_seed_count = cfg.particle_count.saturating_sub(active_seed_count);
    let newly_activated_fraction = newly_activated_count as f32 / inactive_seed_count.max(1) as f32;
    let final_active_mean_radius = final_active_radius_sum / final_active_count.max(1) as f32;
    let mean_final_displacement = mean_displacement(&seed_positions, &trace.positions);
    let max_motion_per_step = trace.mean_dx.iter().copied().fold(0.0_f32, f32::max);
    let target = match case.target {
        GrowthTarget::Torus => TriangleMeshTarget::torus(
            case.seed_scale,
            case.seed_scale * UV_TORUS_MINOR_RATIO,
            64,
            48,
        )
        .unwrap(),
        GrowthTarget::Teapot => TriangleMeshTarget::utah_teapot(case.seed_scale).unwrap(),
    };
    let temporal_progressive_activation = temporal_progressive_activation_report(
        &model,
        &grid,
        &cfg,
        case.seed_mode,
        active_seed_count,
    );
    let front_coherence = front_coherence_report(
        &model,
        &grid,
        &cfg,
        case.seed_mode,
        &trace,
        &seed_positions,
        &seed_states,
    );
    let temporal_geometry_progressive =
        temporal_geometry_progressive_report(&model, &grid, &cfg, case.seed_mode, &target);
    let initial_surface = mesh_surface_stats(&seed_positions, &target);
    let final_surface = mesh_surface_stats(&trace.positions, &target);
    let initial_active_surface = mesh_active_surface_stats(
        &seed_positions,
        &seed_states,
        model.config.state_dims,
        &target,
    );
    let final_active_surface =
        mesh_active_surface_stats(&trace.positions, &trace.states, trace.state_dims, &target);
    let coverage_threshold = target_coverage_threshold(case.seed_scale);
    let initial_target_coverage = target_coverage_stats(
        &seed_positions,
        &target,
        case.particle_count.max(512),
        coverage_threshold,
    );
    let final_target_coverage = target_coverage_stats(
        &trace.positions,
        &target,
        case.particle_count.max(512),
        coverage_threshold,
    );
    let render = mesh_multiview_render_loss_from_trace(
        &trace,
        &target,
        RenderLossConfig {
            image_size: 32,
            target_samples: case.particle_count.max(512),
            world_scale: case.seed_scale * 2.0,
            ..RenderLossConfig::default()
        },
    )
    .unwrap();
    let strict_passed = !position_features
        && non_opacity_seed_abs_max <= 1.0e-6
        && active_seed_count > 0
        && active_seed_count < case.particle_count / 8
        && final_active_count > active_seed_count * 4
        && newly_activated_fraction >= 0.50
        && temporal_progressive_activation
        && front_coherence.passed
        && temporal_geometry_progressive.passed
        && final_active_max_radius > growth_3d_seed_radius(case.seed_scale)
        && max_motion_per_step > 0.01
        && mean_final_displacement > growth_3d_seed_radius(case.seed_scale)
        && final_active_surface.mean < initial_active_surface.mean * 0.85
        && final_active_surface.max < 0.36
        && final_target_coverage.mean < initial_target_coverage.mean * 0.85
        && final_target_coverage.max < case.seed_scale
        && final_target_coverage.covered_fraction >= 0.60
        && render.passed;

    StrictGrowthValidationReport {
        position_features,
        active_seed_count,
        final_active_count,
        newly_activated_count,
        newly_activated_fraction,
        final_active_mean_radius,
        final_active_max_radius,
        non_opacity_seed_abs_max,
        mean_final_displacement,
        initial_surface,
        final_surface,
        initial_active_surface,
        final_active_surface,
        initial_target_coverage,
        final_target_coverage,
        max_motion_per_step,
        render_density_psnr_db: render.density_psnr_db,
        render_color_psnr_db: render.color_psnr_db,
        render_depth_psnr_db: render.depth_psnr_db,
        temporal_progressive_activation,
        temporal_geometry_progressive,
        front_coherence,
        strict_passed,
    }
}

fn color_state_stats(states: &[f32], state_dims: usize) -> ColorStateStats {
    assert!(
        state_dims >= 6,
        "3D growth color validation expects tail color state"
    );
    let tail = state_dims - 3;
    let mut active_count = 0usize;
    let mut active_sum_abs = 0.0_f32;
    let mut active_max_abs = 0.0_f32;
    let mut active_sum = [0.0_f32; 3];
    let mut active_sum_sq = [0.0_f32; 3];

    for state in states.chunks_exact(state_dims) {
        if state[3] <= -1.0 {
            continue;
        }
        active_count += 1;
        let mut particle_max_abs = 0.0_f32;
        for channel in 0..3 {
            let value = state[tail + channel];
            assert!(value.is_finite(), "non-finite tail color state {value}");
            particle_max_abs = particle_max_abs.max(value.abs());
            active_sum[channel] += value;
            active_sum_sq[channel] += value * value;
        }
        active_sum_abs += particle_max_abs;
        active_max_abs = active_max_abs.max(particle_max_abs);
    }

    let mut active_channel_stddev = [0.0_f32; 3];
    if active_count > 0 {
        for channel in 0..3 {
            let mean = active_sum[channel] / active_count as f32;
            let variance = (active_sum_sq[channel] / active_count as f32 - mean * mean).max(0.0);
            active_channel_stddev[channel] = variance.sqrt();
        }
    }

    ColorStateStats {
        active_mean_abs: active_sum_abs / active_count.max(1) as f32,
        active_max_abs,
        active_channel_stddev_mean: active_channel_stddev.iter().sum::<f32>() / 3.0,
    }
}

fn temporal_geometry_progressive_report(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    cfg: &RolloutConfig,
    seed_mode: ParticleSeed,
    target: &TriangleMeshTarget,
) -> MeasuredBool {
    let mut samples = vec![0usize, cfg.steps];
    let mut step = 1usize;
    while step < cfg.steps {
        samples.push(step);
        step *= 2;
    }
    samples.sort_unstable();
    samples.dedup();

    let mut initial = None;
    let mut final_sample = None;
    for steps in samples {
        let (positions, states, state_dims) = if steps == 0 {
            let (positions, states) = seed_particles_scaled(
                cfg.batch_size,
                cfg.particle_count,
                model.config.state_dims,
                model.config.spatial_dims,
                cfg.seed,
                seed_mode,
                cfg.seed_scale,
            );
            (positions, states, model.config.state_dims)
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    steps,
                    ..cfg.clone()
                },
                seed_mode,
            )
            .unwrap();
            (trace.positions, trace.states, trace.state_dims)
        };
        let sample = (
            mesh_active_surface_stats(&positions, &states, state_dims, target),
            target_coverage_stats(
                &positions,
                target,
                cfg.particle_count.max(512),
                target_coverage_threshold(cfg.seed_scale),
            ),
        );
        if steps == 0 {
            initial = Some(sample);
        }
        if steps == cfg.steps {
            final_sample = Some(sample);
        }
    }

    let ((initial_surface, initial_coverage), (final_surface, final_coverage)) =
        match (initial, final_sample) {
            (Some(initial), Some(final_sample)) => (initial, final_sample),
            _ => {
                return MeasuredBool {
                    passed: false,
                    surface_mean_ratio: f32::INFINITY,
                    target_coverage_mean_ratio: f32::INFINITY,
                    target_coverage_fraction_delta: 0.0,
                };
            }
        };
    let surface_mean_ratio = final_surface.mean / initial_surface.mean.max(1.0e-6);
    let target_coverage_mean_ratio = final_coverage.mean / initial_coverage.mean.max(1.0e-6);
    let target_coverage_fraction_delta =
        final_coverage.covered_fraction - initial_coverage.covered_fraction;

    MeasuredBool {
        passed: target_coverage_mean_ratio < 0.85
            && target_coverage_fraction_delta >= 0.10
            && surface_mean_ratio < 0.95,
        surface_mean_ratio,
        target_coverage_mean_ratio,
        target_coverage_fraction_delta,
    }
}

fn temporal_progressive_activation_report(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    cfg: &RolloutConfig,
    seed_mode: ParticleSeed,
    active_seed_count: usize,
) -> bool {
    let mut samples = vec![0usize, cfg.steps];
    let mut step = 1usize;
    while step < cfg.steps {
        samples.push(step);
        step *= 2;
    }
    samples.sort_unstable();
    samples.dedup();

    let mut first_growth_step = None;
    let mut half_activation_step = None;
    let mut full_activation_step = None;
    for steps in samples {
        let active_count = if steps == 0 {
            let (_positions, states) = seed_particles_scaled(
                cfg.batch_size,
                cfg.particle_count,
                model.config.state_dims,
                model.config.spatial_dims,
                cfg.seed,
                seed_mode,
                cfg.seed_scale,
            );
            states
                .chunks_exact(model.config.state_dims)
                .filter(|state| state[3] > -1.0)
                .count()
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    steps,
                    ..cfg.clone()
                },
                seed_mode,
            )
            .unwrap();
            trace
                .states
                .chunks_exact(trace.state_dims)
                .filter(|state| state[3] > -1.0)
                .count()
        };
        let active_fraction = active_count as f32 / cfg.particle_count.max(1) as f32;
        if first_growth_step.is_none()
            && active_count > active_seed_count
            && active_count >= active_seed_count.saturating_mul(2).max(1)
        {
            first_growth_step = Some(steps);
        }
        if half_activation_step.is_none() && active_fraction >= 0.50 {
            half_activation_step = Some(steps);
        }
        if full_activation_step.is_none() && active_fraction >= 0.95 {
            full_activation_step = Some(steps);
        }
    }

    match (
        first_growth_step,
        half_activation_step,
        full_activation_step,
    ) {
        (Some(first), Some(half), Some(full)) => {
            first < half && half < full && full.saturating_sub(first) >= cfg.steps / 4
        }
        _ => false,
    }
}

fn front_coherence_report(
    model: &NpaModel,
    grid: &burn_automata::kernels::HashGridConfig,
    cfg: &RolloutConfig,
    seed_mode: ParticleSeed,
    final_trace: &burn_automata::RolloutTrace,
    seed_positions: &[[f32; 4]],
    seed_states: &[f32],
) -> FrontCoherenceReport {
    let max_allowed_distance = growth_3d_seed_radius(cfg.seed_scale) * 2.5;
    let mut previous: Option<(Vec<[f32; 4]>, Vec<bool>)> = None;
    let mut transition_count = 0usize;
    let mut newly_activated_count = 0usize;
    let mut local_newly_activated_count = 0usize;
    let mut sum_nearest = 0.0_f32;
    let mut max_nearest = 0.0_f32;
    let mut finite = true;

    for steps in temporal_sample_steps(cfg.steps) {
        let (positions, states, state_dims) = if steps == 0 {
            (
                seed_positions.to_vec(),
                seed_states.to_vec(),
                model.config.state_dims,
            )
        } else if steps == cfg.steps {
            (
                final_trace.positions.clone(),
                final_trace.states.clone(),
                final_trace.state_dims,
            )
        } else {
            let trace = run_rollout(
                model,
                grid,
                &RolloutConfig {
                    steps,
                    ..cfg.clone()
                },
                seed_mode,
            )
            .unwrap();
            (trace.positions, trace.states, trace.state_dims)
        };
        let active = active_flags(&states, state_dims);
        if let Some((previous_positions, previous_active)) = previous.take() {
            let previous_active_positions = previous_positions
                .iter()
                .zip(previous_active.iter())
                .filter_map(|(position, active)| (*active).then_some(*position))
                .collect::<Vec<_>>();
            let mut transition_newly_activated = 0usize;
            for idx in 0..active.len() {
                if !active[idx] || previous_active[idx] || previous_active_positions.is_empty() {
                    continue;
                }
                transition_newly_activated += 1;
                newly_activated_count += 1;
                let distance = nearest_distance(positions[idx], &previous_active_positions);
                finite &= distance.is_finite();
                sum_nearest += distance;
                max_nearest = max_nearest.max(distance);
                if distance <= max_allowed_distance {
                    local_newly_activated_count += 1;
                }
            }
            if transition_newly_activated > 0 {
                transition_count += 1;
            }
        }
        previous = Some((positions, active));
    }

    let local_newly_activated_fraction = if newly_activated_count > 0 {
        local_newly_activated_count as f32 / newly_activated_count as f32
    } else {
        0.0
    };
    let mean_nearest_previous_active_distance = if newly_activated_count > 0 {
        sum_nearest / newly_activated_count as f32
    } else {
        f32::INFINITY
    };
    let passed = finite
        && newly_activated_count > 0
        && transition_count >= 2
        && local_newly_activated_fraction >= 0.90
        && mean_nearest_previous_active_distance <= max_allowed_distance * 0.75;

    FrontCoherenceReport {
        passed,
        transition_count,
        newly_activated_count,
        local_newly_activated_fraction,
        mean_nearest_previous_active_distance,
        max_nearest_previous_active_distance: if newly_activated_count > 0 {
            max_nearest
        } else {
            f32::INFINITY
        },
        max_allowed_distance,
    }
}

fn temporal_sample_steps(steps: usize) -> Vec<usize> {
    let mut samples = vec![0usize, steps];
    let mut step = 1usize;
    while step < steps {
        samples.push(step);
        step = step.saturating_mul(2);
        if step == 0 {
            break;
        }
    }
    samples.sort_unstable();
    samples.dedup();
    samples
}

fn active_flags(states: &[f32], state_dims: usize) -> Vec<bool> {
    states
        .chunks_exact(state_dims)
        .map(|state| state_dims > 3 && state[3] > -1.0)
        .collect()
}

fn nearest_distance(position: [f32; 4], candidates: &[[f32; 4]]) -> f32 {
    candidates
        .iter()
        .map(|candidate| {
            ((position[0] - candidate[0]).powi(2)
                + (position[1] - candidate[1]).powi(2)
                + (position[2] - candidate[2]).powi(2))
            .sqrt()
        })
        .fold(f32::INFINITY, f32::min)
}

fn mean_displacement(initial: &[[f32; 4]], final_positions: &[[f32; 4]]) -> f32 {
    initial
        .iter()
        .zip(final_positions.iter())
        .map(|(a, b)| {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        })
        .sum::<f32>()
        / initial.len().max(1) as f32
}

fn mse(model: &NpaModel, batch: &SupervisedBatch) -> f32 {
    let (dx, ds) = model.forward_from_features(&batch.features).unwrap();
    let mut output = Vec::with_capacity(dx.len() * model.config.update_dims());
    for (row, delta) in dx.iter().enumerate() {
        output.extend_from_slice(&delta[..model.config.spatial_dims]);
        let base = row * model.config.state_dims;
        output.extend_from_slice(&ds[base..base + model.config.state_dims]);
    }
    output
        .iter()
        .zip(batch.target_update.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f32>()
        / output.len() as f32
}

#[derive(Clone, Copy, Debug)]
enum GradientParam {
    W1(usize),
    B1(usize),
    W2(usize),
    B2(usize),
}

fn analytic_gradient(grads: &burn_automata::SupervisedGradients, param: GradientParam) -> f32 {
    match param {
        GradientParam::W1(index) => grads.w1[index],
        GradientParam::B1(index) => grads.b1[index],
        GradientParam::W2(index) => grads.w2[index],
        GradientParam::B2(index) => grads.b2[index],
    }
}

fn finite_difference_gradient(
    model: &NpaModel,
    batch: &SupervisedBatch,
    param: GradientParam,
) -> f32 {
    let eps = 1.0e-3;
    let mut plus = model.clone();
    perturb_param(&mut plus, param, eps);
    let mut minus = model.clone();
    perturb_param(&mut minus, param, -eps);
    (supervised_loss(&plus, batch).unwrap() - supervised_loss(&minus, batch).unwrap()) / (2.0 * eps)
}

fn perturb_param(model: &mut NpaModel, param: GradientParam, delta: f32) {
    match param {
        GradientParam::W1(index) => model.weights.w1[index] += delta,
        GradientParam::B1(index) => model.weights.b1[index] += delta,
        GradientParam::W2(index) => model.weights.w2[index] += delta,
        GradientParam::B2(index) => model.weights.b2[index] += delta,
    }
}

#[derive(Clone, Copy, Debug)]
struct SurfaceStats {
    mean: f32,
    max: f32,
}

#[derive(Clone, Copy, Debug)]
struct TargetCoverageStats {
    mean: f32,
    max: f32,
    covered_fraction: f32,
}

fn mesh_surface_stats(positions: &[[f32; 4]], target: &TriangleMeshTarget) -> SurfaceStats {
    surface_stats(positions, |position| {
        target
            .project([position[0], position[1], position[2]])
            .signed_distance
            .abs()
    })
}

fn mesh_active_surface_stats(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    target: &TriangleMeshTarget,
) -> SurfaceStats {
    let mut max = 0.0_f32;
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    for (idx, position) in positions.iter().enumerate() {
        if state_dims <= 3 || states[idx * state_dims + 3] <= -1.0 {
            continue;
        }
        let distance = target
            .project([position[0], position[1], position[2]])
            .signed_distance
            .abs();
        max = max.max(distance);
        sum += distance;
        count += 1;
    }
    SurfaceStats {
        mean: if count > 0 {
            sum / count as f32
        } else {
            f32::INFINITY
        },
        max: if count > 0 { max } else { f32::INFINITY },
    }
}

fn target_coverage_threshold(seed_scale: f32) -> f32 {
    (seed_scale.max(1.0e-4) * 0.18).max(0.04)
}

fn target_coverage_stats(
    positions: &[[f32; 4]],
    target: &TriangleMeshTarget,
    samples: usize,
    threshold: f32,
) -> TargetCoverageStats {
    let samples = samples.max(1);
    let mut sum = 0.0_f32;
    let mut max = 0.0_f32;
    let mut covered = 0usize;
    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        let mut best_distance2 = f32::MAX;
        for position in positions {
            let dx = sample.position[0] - position[0];
            let dy = sample.position[1] - position[1];
            let dz = sample.position[2] - position[2];
            best_distance2 = best_distance2.min(dx * dx + dy * dy + dz * dz);
        }
        let distance = best_distance2.sqrt();
        assert!(distance.is_finite());
        sum += distance;
        max = max.max(distance);
        if distance <= threshold {
            covered += 1;
        }
    }
    TargetCoverageStats {
        mean: sum / samples as f32,
        max,
        covered_fraction: covered as f32 / samples as f32,
    }
}

fn surface_stats(positions: &[[f32; 4]], mut error: impl FnMut([f32; 4]) -> f32) -> SurfaceStats {
    let mut sum = 0.0_f32;
    let mut max = 0.0_f32;
    for position in positions {
        let value = error(*position);
        assert!(value.is_finite());
        sum += value;
        max = max.max(value);
    }
    SurfaceStats {
        mean: sum / positions.len().max(1) as f32,
        max,
    }
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("burn_automata_{}_{}", std::process::id(), name))
}

fn workspace_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn write_fake_pytorch_checkpoint(path: &Path) {
    let file = fs::File::create(path).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    zip.start_file("fake/byteorder", options).unwrap();
    zip.write_all(b"little").unwrap();
    zip.start_file("fake/data/0", options).unwrap();
    write_f32s(&mut zip, &[0.1]);
    zip.start_file("fake/data/1", options).unwrap();
    write_f32s(&mut zip, &[0.5]);
    zip.start_file("fake/data/2", options).unwrap();
    write_f32s(&mut zip, &[0.01; 12]);
    zip.start_file("fake/data/3", options).unwrap();
    write_f32s(&mut zip, &[0.0; 2]);
    zip.start_file("fake/data/4", options).unwrap();
    write_f32s(&mut zip, &[0.02; 6]);
    zip.finish().unwrap();
}

fn write_f32s<W: Write>(writer: &mut W, values: &[f32]) {
    for value in values {
        writer.write_all(&value.to_le_bytes()).unwrap();
    }
}
