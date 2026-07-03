use super::*;
use burn_automata::{
    CONDITION_FEATURE_DIMS, CONDITION_TOKEN_FEATURE_DIMS, ConditionImage2d, HyperAdapterExample2d,
    HyperFlowExample2d, HyperNpa2d, HyperNpa2dConfig, ParticlePriorConfig,
    condition_feature_dims_for_token_grid, generate_conditioned_npa_2d,
    hyper_adapter_regression_loss, hyper_adapter_regression_train_step, hyper_rectified_flow_loss,
    hyper_rectified_flow_train_step,
};

fn small_2d_config() -> NpaConfig {
    NpaConfig {
        state_dims: 2,
        hidden_dims: 4,
        ..NpaConfig::growing_2d()
    }
}

fn condition_image() -> ConditionImage2d {
    ConditionImage2d::from_luma(2, 2, vec![0.0, 1.0, 1.0, 0.0]).unwrap()
}

fn hyper_config() -> HyperNpa2dConfig {
    HyperNpa2dConfig {
        hidden_dims: 4,
        adapter_rank: 1,
        output_scale: 1.0,
        ..HyperNpa2dConfig::default()
    }
}

#[test]
fn condition_summary_tokens_and_prior_are_stable() {
    let image = condition_image();
    let summary = image.summary().unwrap();
    let features = image.feature_vector().unwrap();
    let tokens = image.pooled_tokens(2, 2).unwrap();
    let prior = burn_automata::ParticlePrior2d::from_condition(
        &small_2d_config(),
        &image,
        ParticlePriorConfig {
            min_particles: 10,
            max_particles: 20,
            min_seed_scale: 0.1,
            max_seed_scale: 0.5,
        },
    )
    .unwrap();

    assert_eq!(features.len(), CONDITION_FEATURE_DIMS);
    assert_eq!(tokens.len(), 4);
    assert_eq!(
        image.feature_vector_with_tokens(2, 2).unwrap().len(),
        CONDITION_FEATURE_DIMS + 4 * CONDITION_TOKEN_FEATURE_DIMS
    );
    assert_eq!(
        condition_feature_dims_for_token_grid(2, 2).unwrap(),
        CONDITION_FEATURE_DIMS + 4 * CONDITION_TOKEN_FEATURE_DIMS
    );
    assert!((summary.mean_luma - 0.5).abs() < 1.0e-6);
    assert!((summary.occupancy - 0.5).abs() < 1.0e-6);
    assert_eq!(prior.particle_count, 15);
    assert_eq!(prior.initial_state.len(), small_2d_config().state_dims);
}

#[test]
fn hyper_config_supports_legacy_summary_only_condition_features() {
    let config = small_2d_config();
    let hyper = HyperNpa2d::zeros(
        config,
        HyperNpa2dConfig {
            condition_feature_dims: CONDITION_FEATURE_DIMS,
            condition_token_grid_width: 0,
            condition_token_grid_height: 0,
            ..hyper_config()
        },
    )
    .unwrap();

    assert_eq!(
        hyper
            .predict_adapter_vector(&condition_image())
            .unwrap()
            .len(),
        hyper.adapter_parameter_count()
    );
}

#[test]
fn anchored_hyper_condition_emits_zero_adapter_delta() {
    let config = small_2d_config();
    let mut hyper = HyperNpa2d::seeded(config, hyper_config(), 9).unwrap();
    hyper.set_anchor_condition(&condition_image()).unwrap();

    let values = hyper.predict_adapter_vector(&condition_image()).unwrap();

    assert!(values.iter().all(|value| value.abs() <= 1.0e-7));
}

#[test]
fn generated_conditioned_npa_materializes_valid_adapter_and_model() {
    let config = small_2d_config();
    let base = NpaModel {
        weights: NpaWeights::zeros(&config),
        config,
    };
    let hyper = HyperNpa2d::zeros(base.config.clone(), hyper_config()).unwrap();

    let conditioned = generate_conditioned_npa_2d(
        &base,
        &hyper,
        &condition_image(),
        ParticlePriorConfig {
            min_particles: 8,
            max_particles: 12,
            min_seed_scale: 0.1,
            max_seed_scale: 0.2,
        },
    )
    .unwrap();

    conditioned.adapter.validate(&base.config).unwrap();
    conditioned.model.validate().unwrap();
    assert_eq!(conditioned.adapter.rank, 1);
    assert_eq!(conditioned.model.weights.w1, base.weights.w1);
    assert!((8..=12).contains(&conditioned.prior.particle_count));
}

#[test]
fn hyper_adapter_regression_step_reduces_oracle_adapter_loss() {
    let config = small_2d_config();
    let mut hyper = HyperNpa2d::zeros(config.clone(), hyper_config()).unwrap();
    let mut target = NpaLowRankAdapter::zeros(&config, 1, 2.0);
    target.b2_delta[0] = 0.5;
    let examples = vec![HyperAdapterExample2d {
        condition: condition_image(),
        target_adapter: target,
    }];

    let before = hyper_adapter_regression_loss(&hyper, &examples).unwrap();
    for _ in 0..8 {
        hyper_adapter_regression_train_step(
            &mut hyper,
            &examples,
            SgdConfig {
                learning_rate: 2.0,
                weight_decay: 0.0,
                grad_clip_norm: 0.0,
            },
        )
        .unwrap();
    }
    let after = hyper_adapter_regression_loss(&hyper, &examples).unwrap();
    let predicted = hyper.predict_adapter(&condition_image()).unwrap();

    assert!(after < before, "after={after} before={before}");
    assert!(predicted.b2_delta[0] > 0.0);
}

#[test]
fn hyper_rectified_flow_step_reduces_teacher_update_loss() {
    let config = small_2d_config();
    let base = NpaModel {
        weights: NpaWeights::zeros(&config),
        config,
    };
    let mut hyper = HyperNpa2d::zeros(base.config.clone(), hyper_config()).unwrap();
    let mut target_update = vec![0.0; base.config.update_dims()];
    target_update[0] = 0.5;
    let examples = vec![HyperFlowExample2d {
        condition: condition_image(),
        batch: SupervisedBatch {
            features: vec![0.0; base.config.perception_dims()],
            target_update,
        },
    }];

    let before = hyper_rectified_flow_loss(&base, &hyper, &examples).unwrap();
    let report = hyper_rectified_flow_train_step(
        &base,
        &mut hyper,
        &examples,
        SgdConfig {
            learning_rate: 0.2,
            weight_decay: 0.0,
            grad_clip_norm: 0.0,
        },
    )
    .unwrap();
    let after = hyper_rectified_flow_loss(&base, &hyper, &examples).unwrap();
    let predicted = hyper.predict_adapter(&condition_image()).unwrap();

    assert_eq!(report.rows, 1);
    assert!(report.grad_norm > 0.0);
    assert!(after < before, "after={after} before={before}");
    assert!(predicted.b2_delta[0] > 0.0);
}
