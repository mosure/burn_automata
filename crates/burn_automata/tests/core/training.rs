use super::*;

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
fn rollout_supervision_temporal_samples_cover_multiple_snapshots() {
    let (mut config, grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
    config.hidden_dims = 8;
    let teacher = NpaModel::seeded(config.clone(), 31);
    let student = NpaModel::seeded(config.clone(), 7);
    let terminal = rollout_supervised_batch_from_model(
        &student,
        &teacher,
        &grid,
        SupervisedTarget::Teacher(&teacher),
        RolloutSupervisionConfig {
            max_rows: 18,
            particle_count: 36,
            rollout_steps: 4,
            rollouts: 1,
            temporal_samples: 1,
            update_prob: 0.5,
            seed: 99,
            seed_scale: NpaConfig::seed_scale_for_preset(AutomataPreset::Growing2d),
            seed_mode: ParticleSeed::UniformCircle,
            ..RolloutSupervisionConfig::default()
        },
    )
    .unwrap();
    let temporal = rollout_supervised_batch_from_model(
        &student,
        &teacher,
        &grid,
        SupervisedTarget::Teacher(&teacher),
        RolloutSupervisionConfig {
            max_rows: 18,
            particle_count: 36,
            rollout_steps: 4,
            rollouts: 1,
            temporal_samples: 3,
            update_prob: 0.5,
            seed: 99,
            seed_scale: NpaConfig::seed_scale_for_preset(AutomataPreset::Growing2d),
            seed_mode: ParticleSeed::UniformCircle,
            ..RolloutSupervisionConfig::default()
        },
    )
    .unwrap();

    assert_eq!(terminal.features.len(), 18 * config.perception_dims());
    assert_eq!(temporal.features.len(), 18 * config.perception_dims());
    assert_eq!(temporal.target_update.len(), 18 * config.update_dims());
    assert!(temporal.features.iter().all(|value| value.is_finite()));
    assert_ne!(terminal.features, temporal.features);
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
