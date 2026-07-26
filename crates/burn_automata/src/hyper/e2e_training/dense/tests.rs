//! Backend-instantiated correctness and throughput regression tests.

use super::*;
use super::adaptive_training::adaptive_backward_scale;

fn test_adaptive_target2d_seed_bank(
    pool_size: usize,
    particle_count: usize,
    state_dims: usize,
    config: DirectBasisTrainConfig,
) -> crate::adaptive::AdaptiveTarget2dSeedBank {
    let update_masks = |rows: usize| {
        (0..rows)
            .flat_map(|_| {
                (0..particle_count).map(|particle| {
                    let mut keys = [0; 6];
                    let mut weights = [0.0; 6];
                    keys[0] = particle as u32;
                    weights[0] = 1.0;
                    crate::adaptive::AdaptiveTarget2dUpdateMask {
                        keys,
                        weights,
                        expected: false,
                    }
                })
            })
            .collect::<Vec<_>>()
    };
    let (positions, states) = seed_particles_scaled(
        pool_size,
        particle_count,
        state_dims,
        2,
        config.seed,
        config.seed_mode,
        0.2,
    );
    let (eval_positions, eval_states) = seed_particles_scaled(
        1,
        particle_count,
        state_dims,
        2,
        config.eval_seed,
        config.seed_mode,
        0.2,
    );
    crate::adaptive::AdaptiveTarget2dSeedBank {
        positions: positions
            .into_iter()
            .flat_map(|position| [position[0], position[1]])
            .collect(),
        states,
        update_masks: update_masks(pool_size),
        eval_positions: eval_positions
            .into_iter()
            .flat_map(|position| [position[0], position[1]])
            .collect(),
        eval_states,
        eval_update_masks: update_masks(1),
        eval_seeds: vec![config.eval_seed],
        pool_size,
        particle_count,
        state_dims,
        max_measure_relative_error: 0.0,
        max_centroid_l2_error: 0.0,
        max_extensive_state_l2_error: 0.0,
    }
}

#[test]
fn e2e_learning_rate_warmup_is_linear_and_bounded() {
    assert_eq!(e2e_lr_warmup_scale(0, 1), 1.0);
    assert_eq!(e2e_lr_warmup_scale(100, 1), 0.01);
    assert_eq!(e2e_lr_warmup_scale(100, 50), 0.5);
    assert_eq!(e2e_lr_warmup_scale(100, 100), 1.0);
    assert_eq!(e2e_lr_warmup_scale(100, 500), 1.0);
}

#[test]
fn adaptive_trajectory_tail_count_is_bounded_and_ceil_rounded() {
    assert_eq!(adaptive_training::adaptive_trajectory_tail_count(0, 0.25), 0);
    assert_eq!(adaptive_training::adaptive_trajectory_tail_count(16, 0.0), 0);
    assert_eq!(adaptive_training::adaptive_trajectory_tail_count(16, 0.01), 1);
    assert_eq!(adaptive_training::adaptive_trajectory_tail_count(16, 0.25), 4);
    assert_eq!(adaptive_training::adaptive_trajectory_tail_count(16, 1.0), 16);
}

#[test]
fn target2d_oracle_schedule_matches_upstream_update_boundaries() {
    let milestones = [2_000, 4_000, 6_000, 8_000];
    assert_eq!(milestone_lr_scale(1, &milestones, 0.3), 1.0);
    assert_eq!(milestone_lr_scale(2_000, &milestones, 0.3), 1.0);
    assert!((milestone_lr_scale(2_001, &milestones, 0.3) - 0.3).abs() < 1.0e-7);
    assert!((milestone_lr_scale(4_001, &milestones, 0.3) - 0.09).abs() < 1.0e-7);
    assert!((milestone_lr_scale(8_001, &milestones, 0.3) - 0.0081).abs() < 1.0e-7);

    assert_eq!(oracle_repetition_position(1, 10_001), (0, 1, 0));
    assert_eq!(
        oracle_repetition_position(10_001, 10_001),
        (0, 10_001, 10_000)
    );
    assert_eq!(oracle_repetition_position(10_002, 10_001), (1, 1, 10_000));
    assert_eq!(
        oracle_repetition_position(20_002, 10_001),
        (1, 10_001, 20_000)
    );
}

#[test]
fn dino_prefetch_deduplicates_rollout_replicas_and_preserves_order() {
    let (unique, expansion) = deduplicate_condition_indices(&[7, 7, 7, 2, 2, 9, 7, 9]);
    assert_eq!(unique, vec![7, 2, 9]);
    assert_eq!(expansion, vec![0, 0, 0, 1, 1, 2, 0, 2]);
    let restored = expansion
        .into_iter()
        .map(|row| unique[row])
        .collect::<Vec<_>>();
    assert_eq!(restored, vec![7, 7, 7, 2, 2, 9, 7, 9]);
}

#[test]
fn warm_start_output_bias_contract_is_explicit() {
    let legacy = validate_warm_start_output_bias_contract(None, false).unwrap_err();
    assert!(
        legacy.to_string().contains("legacy artifacts"),
        "unexpected error: {legacy}"
    );
    validate_warm_start_output_bias_contract(Some(false), false).unwrap();
    validate_warm_start_output_bias_contract(Some(true), true).unwrap();
    assert!(validate_warm_start_output_bias_contract(Some(true), false).is_err());
    assert!(validate_warm_start_output_bias_contract(Some(false), true).is_err());
}

#[test]
fn condition_diagnostics_are_bounded_and_cover_the_dataset() {
    assert!(condition_diagnostic_indices(0).is_empty());
    assert_eq!(condition_diagnostic_indices(1), vec![0]);
    assert_eq!(condition_diagnostic_indices(4), vec![0, 1, 2, 3]);

    let indices = condition_diagnostic_indices(100_000);
    assert_eq!(indices.len(), E2E_CONDITION_DIAGNOSTIC_ROWS);
    assert_eq!(indices.first(), Some(&0));
    assert_eq!(indices.last(), Some(&99_999));
    assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn matching_current_checkpoint_is_promoted_without_model_serialization() {
    let root = std::env::temp_dir().join(format!(
        "burn_automata_e2e_checkpoint_promotion_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("current_shared_base.bpk"), b"base-step-7").unwrap();
    std::fs::write(root.join("current_hyper_2d.bpk"), b"hyper-step-7").unwrap();
    std::fs::write(root.join("current_training_state.mpk"), b"state-step-7").unwrap();
    std::fs::write(
        root.join("current_metadata.json"),
        serde_json::to_vec_pretty(&json!({
            "label": "current",
            "step": 7,
            "source": "current-source",
            "shared_base_output": root.join("current_shared_base.bpk"),
            "shared_base_sha256": "base-hash",
            "hyper_output": root.join("current_hyper_2d.bpk"),
            "hyper_sha256": "hyper-hash",
        }))
        .unwrap(),
    )
    .unwrap();

    let hashes = promote_matching_current_e2e_checkpoint(&root, 7)
        .unwrap()
        .expect("matching current checkpoint should be promoted");
    assert_eq!(hashes.shared_base_sha256, "base-hash");
    assert_eq!(hashes.hyper_sha256, "hyper-hash");
    assert_eq!(
        std::fs::read(root.join("best_shared_base.bpk")).unwrap(),
        b"base-step-7"
    );
    assert_eq!(
        std::fs::read(root.join("best_hyper_2d.bpk")).unwrap(),
        b"hyper-step-7"
    );
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("best_metadata.json")).unwrap()).unwrap();
    assert_eq!(metadata["label"], "best");
    assert_eq!(metadata["step"], 7);
    assert!(
        metadata["source"]
            .as_str()
            .unwrap()
            .contains("label=best:step=7")
    );
    promote_current_e2e_training_checkpoint(&root).unwrap();
    assert_eq!(
        std::fs::read(root.join("best_training_state.mpk")).unwrap(),
        b"state-step-7"
    );
    assert_eq!(
        e2e_training_checkpoint_artifact_label(&root.join("current_training_state.mpk")),
        "current"
    );
    assert_eq!(
        e2e_training_checkpoint_artifact_label(&root.join("best_training_state.mpk")),
        "best"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn matching_best_checkpoint_bundle_is_reusable_without_reserialization() {
    let root = std::env::temp_dir().join(format!(
        "burn_automata_e2e_best_checkpoint_reuse_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("best_shared_base.bpk"), b"base-step-7").unwrap();
    std::fs::write(root.join("best_hyper_2d.bpk"), b"hyper-step-7").unwrap();
    std::fs::write(root.join("best_training_state.mpk"), b"state-step-7").unwrap();
    std::fs::write(
        root.join("best_metadata.json"),
        serde_json::to_vec_pretty(&json!({
            "label": "best",
            "step": 7,
            "shared_base_sha256": "base-hash-7",
            "hyper_sha256": "hyper-hash-7",
        }))
        .unwrap(),
    )
    .unwrap();

    let hashes = matching_e2e_checkpoint_artifacts(&root, "best", 7)
        .unwrap()
        .expect("matching best bundle should be reusable");
    assert_eq!(hashes.shared_base_sha256, "base-hash-7");
    assert_eq!(hashes.hyper_sha256, "hyper-hash-7");
    assert!(
        matching_e2e_checkpoint_artifacts(&root, "best", 8)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        std::fs::read(root.join("best_training_state.mpk")).unwrap(),
        b"state-step-7"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn external_resume_checkpoint_is_promoted_as_local_incumbent() {
    let root = std::env::temp_dir().join(format!(
        "burn_automata_promote_resume_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let source_dir = root.join("source");
    let destination_dir = root.join("destination");
    std::fs::create_dir_all(&source_dir).unwrap();
    let source = source_dir.join("best_training_state.mpk");
    std::fs::write(source_dir.join("best_shared_base.bpk"), b"resume-base").unwrap();
    std::fs::write(source_dir.join("best_hyper_2d.bpk"), b"resume-hyper").unwrap();
    std::fs::write(
        source_dir.join("best_metadata.json"),
        serde_json::to_vec_pretty(&json!({
            "label": "best",
            "shared_base_output": source_dir.join("best_shared_base.bpk"),
            "hyper_output": source_dir.join("best_hyper_2d.bpk"),
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(&source, b"resume-state").unwrap();

    promote_resume_e2e_checkpoint(&source, &destination_dir).unwrap();

    assert_eq!(
        std::fs::read(destination_dir.join("best_training_state.mpk")).unwrap(),
        b"resume-state"
    );
    assert_eq!(
        std::fs::read(destination_dir.join("best_shared_base.bpk")).unwrap(),
        b"resume-base"
    );
    assert_eq!(
        std::fs::read(destination_dir.join("best_hyper_2d.bpk")).unwrap(),
        b"resume-hyper"
    );
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(destination_dir.join("best_metadata.json")).unwrap())
            .unwrap();
    assert_eq!(metadata["label"], "best");
    assert_eq!(
        metadata["shared_base_output"],
        json!(destination_dir.join("best_shared_base.bpk"))
    );
    assert!(!destination_dir.join("current_training_state.mpk").exists());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn selected_final_checkpoint_promotes_matching_training_state() {
    let root = std::env::temp_dir().join(format!(
        "burn_automata_promote_final_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("current_training_state.mpk"), b"final-state").unwrap();
    std::fs::write(root.join("best_training_state.mpk"), b"prior-state").unwrap();

    assert!(!promote_selected_final_e2e_training_checkpoint(9, 10, &root).unwrap());
    assert_eq!(
        std::fs::read(root.join("best_training_state.mpk")).unwrap(),
        b"prior-state"
    );
    assert!(promote_selected_final_e2e_training_checkpoint(10, 10, &root).unwrap());
    assert_eq!(
        std::fs::read(root.join("best_training_state.mpk")).unwrap(),
        b"final-state"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn amortization_mix_keeps_explicit_hyper_only_trajectories() {
    assert_eq!(
        e2e_amortization_mix_scales(8, 4, 0.25, 0.8),
        vec![0.0, 0.8, 0.8, 0.8, 0.0, 0.8, 0.8, 0.8]
    );
    assert_eq!(e2e_amortization_mix_scales(4, 4, 0.0, 0.8), vec![0.8; 4]);
    assert_eq!(e2e_amortization_mix_scales(4, 4, 1.0, 0.8), vec![0.0; 4]);
    let identities = [3, 3, 3, 3, 7, 7, 7, 7];
    assert_eq!(
        e2e_amortization_active_identities(&identities, 4, 0.25, 0.8, false),
        vec![3, 3, 3, 7, 7, 7]
    );
    assert!(e2e_amortization_active_identities(&identities, 4, 1.0, 0.8, false).is_empty());
    assert_eq!(
        e2e_amortization_active_identities(&identities, 4, 0.25, 0.0, false),
        Vec::<usize>::new()
    );
    assert_eq!(
        e2e_amortization_active_identities(&identities, 4, 1.0, 0.0, true),
        identities
    );
}

#[test]
fn curriculum_resume_requires_exact_training_identity_order() {
    let recorded = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    assert!(curriculum_source_order_matches(&recorded, &recorded));
    assert!(!curriculum_source_order_matches(
        &recorded,
        &["a".to_string(), "c".to_string(), "b".to_string()]
    ));
    assert!(!curriculum_source_order_matches(
        &recorded,
        &["a".to_string(), "b".to_string()]
    ));
}

#[test]
fn nonfinite_loss_diagnostic_preserves_all_component_values() {
    let device = BurnDevice::default();
    let scalar =
        |value| Tensor::<BurnBackend, 1>::from_data(TensorData::new(vec![value], [1]), &device);
    let error = loss_vector_scalars(BurnLossBatchTensors {
        total: scalar(f32::INFINITY),
        splat: scalar(0.25),
        color: scalar(f32::NAN),
        density: scalar(0.5),
    })
    .err()
    .expect("non-finite loss should fail")
    .to_string();
    assert!(error.contains("total=inf"));
    assert!(error.contains("splat=0.25"));
    assert!(error.contains("color=NaN"));
    assert!(error.contains("density=0.5"));
}

#[test]
fn finite_value_summary_counts_nan_and_infinities() {
    let summary = finite_values_summary(
        "state",
        &[0.0, -2.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
    );
    assert_eq!(
        summary,
        "state[len=5 finite=2 nan=1 +inf=1 -inf=1 min=-2.000000e0 max=0.000000e0]"
    );
}

#[test]
fn optimizer_boundary_sanitizes_nonfinite_gradients_on_device() {
    let device = BurnDevice::default();
    let inner_device: Device<InnerBackend> = device;
    let mut gradients = vec![Tensor::<InnerBackend, 2>::from_data(
        TensorData::new(
            vec![1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            [2, 2],
        ),
        &inner_device,
    )];
    sanitize_nonfinite_gradients(&mut gradients);
    assert_eq!(
        tensor_vec(gradients.remove(0)).unwrap(),
        [1.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn model_batch_optimizer_boundary_sanitizes_nonfinite_gradients_on_device() {
    let device = BurnDevice::default();
    let inner_device: Device<InnerBackend> = device;
    let mut gradients = vec![Tensor::<InnerBackend, 3>::from_data(
        TensorData::new(
            vec![1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            [1, 2, 2],
        ),
        &inner_device,
    )];
    sanitize_nonfinite_model_batch_gradients(&mut gradients);
    assert_eq!(
        tensor3_vec(gradients.remove(0)).unwrap(),
        [1.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn model_batch_optimizer_normalizes_large_finite_gradients_without_overflow() {
    let device = BurnDevice::default();
    let inner_device: Device<InnerBackend> = device;
    let mut gradients = vec![Tensor::<InnerBackend, 3>::from_data(
        TensorData::new(vec![1.0e30, -1.0e30, 1.0e30, -1.0e30], [1, 2, 2]),
        &inner_device,
    )];
    let (norms, _, _) =
        prepare_model_batch_grad_group(&mut gradients, 0.0, true, true).unwrap();
    assert!(norms[0].is_finite());
    assert!((norms[0] / 2.0e30 - 1.0).abs() <= 1.0e-5);
    let normalized = tensor3_vec(gradients.remove(0)).unwrap();
    assert!(normalized.iter().all(|value| value.is_finite()));
    let normalized_norm = normalized.iter().map(|value| value * value).sum::<f32>().sqrt();
    assert!((normalized_norm - 1.0).abs() <= 1.0e-5);
}

#[test]
fn model_batch_scale_only_optimizer_preserves_native_rule_parameters() {
    let device = BurnDevice::default();
    let mut config = NpaConfig::growing_2d();
    config.auxiliary_input_dims = 1;
    let model = NpaModel::seeded(config, 17);
    let initial = model.weights.clone();
    let mut params = BurnBaseBatch::from_models(std::slice::from_ref(&model), &device).unwrap();
    let mut optimizer = BurnBaseBatchAdamWState::zeros_like(&params);
    let mut grads = (params.w1.clone().sum()
        + params.b1.clone().sum()
        + params.w2.clone().sum()
        + params.b2.clone().sum())
    .backward();
    params
        .apply_adamw_last_input_column(
            &mut grads,
            &mut optimizer,
            AdamWConfig {
                learning_rate: 1.0e-3,
                weight_decay: 0.0,
                ..AdamWConfig::default()
            },
            true,
            true,
        )
        .unwrap();
    let mut updated = model.clone();
    params
        .write_to_models(std::slice::from_mut(&mut updated))
        .unwrap();

    let input_dims = model.config.perception_dims();
    for hidden in 0..model.config.hidden_dims {
        let row = hidden * input_dims;
        assert_eq!(
            &updated.weights.w1[row..row + input_dims - 1],
            &initial.w1[row..row + input_dims - 1],
        );
        assert_ne!(
            updated.weights.w1[row + input_dims - 1],
            initial.w1[row + input_dims - 1],
        );
    }
    assert_eq!(updated.weights.b1, initial.b1);
    assert_eq!(updated.weights.w2, initial.w2);
    assert_eq!(updated.weights.b2, initial.b2);
}

#[test]
fn tbptt_gradient_chunks_apply_one_optimizer_update() {
    let device = BurnDevice::default();
    let model = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 17);
    let mut params = BurnBaseParams::from_model(&model, &device).unwrap();
    let mut optimizer = BurnBaseAdamWState::zeros_like(&params);
    let mut accumulated = None;

    for scale in [0.25, 0.75, 1.25] {
        let mut grads = (params.w1.clone().sum()
            + params.b1.clone().sum()
            + params.w2.clone().sum()
            + params.b2.clone().sum())
        .mul_scalar(scale)
        .backward();
        accumulate_gradient_group(&mut accumulated, params.take_gradients(&mut grads));
    }

    let config = AdamWConfig {
        learning_rate: 1.0e-4,
        weight_decay: 0.0,
        grad_clip_norm: 0.0,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1.0e-8,
    };
    params
        .apply_adamw_gradients(
            accumulated.expect("TBPTT chunks should produce gradients"),
            &mut optimizer,
            config,
            false,
            true,
        )
        .unwrap();

    assert_eq!(optimizer.step, 1);
}

#[test]
fn final_only_tbptt_reserves_a_full_gradient_suffix() {
    let chunks = |rollout_steps, chunk_steps| {
        let mut remaining = rollout_steps;
        let mut chunks = Vec::new();
        while remaining > 0 {
            let steps = tbptt_next_chunk_steps(remaining, chunk_steps, true);
            chunks.push((steps, remaining <= chunk_steps));
            remaining -= steps;
        }
        chunks
    };

    assert_eq!(chunks(32, 64), vec![(32, true)]);
    assert_eq!(chunks(64, 64), vec![(64, true)]);
    assert_eq!(chunks(65, 64), vec![(1, false), (64, true)]);
    assert_eq!(chunks(95, 64), vec![(31, false), (64, true)]);
    assert_eq!(
        chunks(150, 64),
        vec![(64, false), (22, false), (64, true)]
    );
}

#[test]
fn sample_id_table_gradients_normalize_each_parameter_and_identity() {
    let device = BurnDevice::default();
    let inner_device: Device<InnerBackend> = device;
    let gradient = Tensor::<InnerBackend, 2>::from_data(
        TensorData::new(vec![3.0, 0.0, 4.0, 5.0, 0.0, 6.0, 0.0, 8.0], [4, 2]),
        &inner_device,
    );
    let normalized = normalize_sample_id_table_gradient(gradient, &[(0, 2), (2, 2)]);
    let values = normalized.into_data().to_vec::<f32>().unwrap();
    let expected = [0.6, 0.0, 0.8, 1.0, 0.0, 0.6, 0.0, 0.8];
    assert!(max_abs_difference(&values, &expected) <= 1.0e-5);
}

#[test]
fn dense_endpoint_gradients_match_upstream_parameter_groups() {
    let device = BurnDevice::default();
    let inner_device: Device<InnerBackend> = device;
    let config = NpaConfig::growing_2d();
    let layout = NpaParameterRowLayout2d::new(&config);
    let identities = 2;
    let row_dims = layout.max_row_dims();
    let packed_values = layout.row_count() * row_dims;
    let mut values = vec![0.0_f32; packed_values * identities];
    let mut set = |row: usize, column: usize, identity: usize, value: f32| {
        values[(row * row_dims + column) * identities + identity] = value;
    };
    set(0, 0, 0, 3.0);
    set(0, 1, 0, 4.0);
    set(0, config.perception_dims(), 0, 5.0);
    set(config.hidden_dims, 0, 0, 6.0);
    set(config.hidden_dims, 1, 0, 8.0);
    set(1, 0, 1, 2.0);
    set(1, config.perception_dims(), 1, 3.0);
    set(config.hidden_dims + 1, 0, 1, 4.0);
    set(0, config.perception_dims() + 1, 0, 99.0);
    set(config.hidden_dims, config.hidden_dims, 0, 99.0);

    let gradient = Tensor::<InnerBackend, 2>::from_data(
        TensorData::new(values, [packed_values, identities]),
        &inner_device,
    );
    let normalized =
        normalize_packed_npa_table_gradient(gradient, PackedNpaGradientLayout::new(&config, false));
    let values = normalized.into_data().to_vec::<f32>().unwrap();
    let get = |row: usize, column: usize, identity: usize| {
        values[(row * row_dims + column) * identities + identity]
    };
    assert!((get(0, 0, 0) - 0.6).abs() <= 1.0e-5);
    assert!((get(0, 1, 0) - 0.8).abs() <= 1.0e-5);
    assert!((get(0, config.perception_dims(), 0) - 1.0).abs() <= 1.0e-5);
    assert!((get(config.hidden_dims, 0, 0) - 0.6).abs() <= 1.0e-5);
    assert!((get(config.hidden_dims, 1, 0) - 0.8).abs() <= 1.0e-5);
    assert!((get(1, 0, 1) - 1.0).abs() <= 1.0e-5);
    assert!((get(1, config.perception_dims(), 1) - 1.0).abs() <= 1.0e-5);
    assert!((get(config.hidden_dims + 1, 0, 1) - 1.0).abs() <= 1.0e-5);
    assert_eq!(get(0, config.perception_dims() + 1, 0), 0.0);
    assert_eq!(get(config.hidden_dims, config.hidden_dims, 0), 0.0);
}

#[test]
fn sparse_table_adamw_freezes_absent_identities_and_steps_active_once() {
    let device = BurnDevice::default();
    let inner_device: Device<InnerBackend> = device;
    let param = Tensor::<InnerBackend, 2>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
        &inner_device,
    );
    let gradient = Tensor::<InnerBackend, 2>::from_data(
        TensorData::new(vec![0.0, 2.0, 0.0, 0.0, -4.0, 0.0], [2, 3]),
        &inner_device,
    );
    let mut moment = Tensor::<InnerBackend, 2>::from_data(
        TensorData::new(vec![0.2, 0.0, 0.4, 0.5, 0.0, 0.7], [2, 3]),
        &inner_device,
    );
    let mut velocity = Tensor::<InnerBackend, 2>::from_data(
        TensorData::new(vec![0.3, 0.0, 0.5, 0.6, 0.0, 0.8], [2, 3]),
        &inner_device,
    );
    let original_moment = tensor_vec(moment.clone()).unwrap();
    let original_velocity = tensor_vec(velocity.clone()).unwrap();
    let mut identity_steps = vec![0; 3];
    let config = AdamWConfig {
        learning_rate: 0.1,
        weight_decay: 0.2,
        grad_clip_norm: 0.0,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1.0e-8,
    };
    let updated = apply_sparse_column_adamw_tensor(
        param,
        gradient,
        &mut moment,
        &mut velocity,
        config,
        Tensor::<InnerBackend, 1>::ones([1], &inner_device),
        SparseIdentityAdamW {
            identity_steps: &mut identity_steps,
            active_identities: &[1, 1],
            upstream_growing_min_lr_scale: None,
        },
    )
    .unwrap();

    assert_eq!(identity_steps, [0, 1, 0]);
    let updated = tensor_vec(updated).unwrap();
    assert_eq!(
        [updated[0], updated[2], updated[3], updated[5]],
        [1.0, 3.0, 4.0, 6.0]
    );
    let moment = tensor_vec(moment).unwrap();
    let velocity = tensor_vec(velocity).unwrap();
    for index in [0, 2, 3, 5] {
        assert_eq!(moment[index], original_moment[index]);
        assert_eq!(velocity[index], original_velocity[index]);
    }
    assert!((updated[1] - 1.86).abs() < 1.0e-5);
    assert!((updated[4] - 5.0).abs() < 1.0e-5);
}

#[test]
fn upstream_growing_sparse_schedule_is_per_identity_and_resets_moments() {
    assert_eq!(upstream_growing_identity_schedule(1, 0.0), (1, 1.0, false));
    assert_eq!(
        upstream_growing_identity_schedule(2_001, 0.0),
        (2_001, 0.3, false)
    );
    assert_eq!(
        upstream_growing_identity_schedule(10_001, 0.0),
        (1, 1.0, true)
    );
    assert!(
        (mean_upstream_growing_identity_lr_scale(&[0, 2_000, 4_000], &[0, 0, 1, 1], 0.0,) - 0.65)
            .abs()
            < 1.0e-6
    );

    let device = BurnDevice::default();
    let inner_device: Device<InnerBackend> = device;
    let param = Tensor::<InnerBackend, 2>::from_data(
        TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]),
        &inner_device,
    );
    let gradient = Tensor::<InnerBackend, 2>::from_data(
        TensorData::new(vec![0.0, 2.0, 0.0, -4.0], [2, 2]),
        &inner_device,
    );
    let mut moment = Tensor::<InnerBackend, 2>::full([2, 2], 9.0, &inner_device);
    let mut velocity = Tensor::<InnerBackend, 2>::full([2, 2], 7.0, &inner_device);
    let mut identity_steps = vec![0, 10_000];
    let updated = apply_sparse_column_adamw_tensor(
        param,
        gradient,
        &mut moment,
        &mut velocity,
        AdamWConfig {
            learning_rate: 0.1,
            weight_decay: 0.0,
            grad_clip_norm: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
        },
        Tensor::<InnerBackend, 1>::ones([1], &inner_device),
        SparseIdentityAdamW {
            identity_steps: &mut identity_steps,
            active_identities: &[1],
            upstream_growing_min_lr_scale: Some(0.0),
        },
    )
    .unwrap();

    assert_eq!(identity_steps, [0, 10_001]);
    let updated = tensor_vec(updated).unwrap();
    assert!((updated[1] - 1.9).abs() < 1.0e-5);
    assert!((updated[3] - 4.1).abs() < 1.0e-5);
    let moment = tensor_vec(moment).unwrap();
    let velocity = tensor_vec(velocity).unwrap();
    assert!((moment[1] - 0.2).abs() < 1.0e-5);
    assert!((moment[3] + 0.4).abs() < 1.0e-5);
    assert!((velocity[1] - 0.004).abs() < 1.0e-6);
    assert!((velocity[3] - 0.016).abs() < 1.0e-6);
}

#[test]
fn batched_gemm_oracle_models_match_independent_forward_and_adamw() {
    let device = BurnDevice::default();
    let config = NpaConfig::growing_2d();
    let source_models = [
        NpaModel::upstream_seeded(config.clone(), 41),
        NpaModel::upstream_seeded(config.clone(), 97),
    ];
    let repeats = 3;
    let particles = 5;
    let feature_dims = config.perception_dims();
    let feature_values = (0..source_models.len() * repeats * particles * feature_dims)
        .map(|index| ((index % 37) as f32 - 18.0) * 0.003)
        .collect::<Vec<_>>();
    let features = tensor3(
        feature_values,
        [source_models.len() * repeats, particles, feature_dims],
        &device,
    );

    let mut batch = BurnBaseBatch::from_models(&source_models, &device).unwrap();
    let batched_output = batch.forward(features.clone());
    let mut serial_outputs = Vec::new();
    let mut serial_params = Vec::new();
    for (model, source_model) in source_models.iter().enumerate() {
        let params = BurnBaseParams::from_model(source_model, &device).unwrap();
        for repeat in 0..repeats {
            let row = model * repeats + repeat;
            let feature = features.clone().narrow(0, row, 1).squeeze_dim::<2>(0);
            let hidden = relu(
                feature.clone().matmul(params.w1.clone().transpose())
                    + params.b1.clone().expand([particles, config.hidden_dims]),
            );
            serial_outputs.push(
                (hidden.matmul(params.w2.clone().transpose())
                    + params.b2.clone().expand([particles, config.update_dims()]))
                .unsqueeze_dim::<3>(0),
            );
        }
        serial_params.push(params);
    }
    let serial_output = Tensor::cat(serial_outputs, 0);
    let serial_values = tensor3_vec(serial_output.clone().inner()).unwrap();
    let batched_values = tensor3_vec(batched_output.inner()).unwrap();
    let forward_diff = max_abs_difference(&batched_values, &serial_values);
    let forward_diff_index = batched_values
        .iter()
        .zip(&serial_values)
        .enumerate()
        .max_by(|(_, (lhs_a, rhs_a)), (_, (lhs_b, rhs_b))| {
            (*lhs_a - *rhs_a).abs().total_cmp(&(*lhs_b - *rhs_b).abs())
        })
        .map(|(index, _)| index)
        .unwrap_or_default();
    assert!(
        forward_diff <= 1.0e-5,
        "batched GEMM forward max diff {forward_diff} at {forward_diff_index}: batched={} serial={}; first batched={:?} serial={:?}",
        batched_values[forward_diff_index],
        serial_values[forward_diff_index],
        &batched_values[..batched_values.len().min(16)],
        &serial_values[..serial_values.len().min(16)],
    );

    let optimizer_config = AdamWConfig {
        learning_rate: 5.0e-4,
        weight_decay: 0.0,
        grad_clip_norm: 0.0,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1.0e-8,
    };
    let repeated = batch.repeated(repeats);
    let batch_loss = (repeated.w1.mul_scalar(0.7).sum()
        + repeated.b1.mul_scalar(-0.2).sum()
        + repeated.w2.mul_scalar(0.3).sum()
        + repeated.b2.mul_scalar(-0.5).sum())
    .div_scalar(repeats as f32);
    let mut batch_grads = batch_loss.backward();
    let mut batch_optimizer = BurnBaseBatchAdamWState::zeros_like(&batch);
    let (batch_norms, batch_scales) = batch
        .apply_adamw(
            &mut batch_grads,
            &mut batch_optimizer,
            optimizer_config,
            true,
            true,
        )
        .unwrap();

    let mut serial_models = source_models.to_vec();
    let mut serial_norms = Vec::new();
    let mut serial_scales = Vec::new();
    for (params, model) in serial_params.iter_mut().zip(&mut serial_models) {
        let loss = params.w1.clone().mul_scalar(0.7).sum()
            + params.b1.clone().mul_scalar(-0.2).sum()
            + params.w2.clone().mul_scalar(0.3).sum()
            + params.b2.clone().mul_scalar(-0.5).sum();
        let mut grads = loss.backward();
        let mut optimizer = BurnBaseAdamWState::zeros_like(params);
        let (norm, scale) = params
            .apply_adamw(&mut grads, &mut optimizer, optimizer_config, true, true)
            .unwrap();
        serial_norms.push(norm);
        serial_scales.push(scale);
        params.write_to_model(model).unwrap();
    }
    let mut batch_models = source_models.to_vec();
    batch.write_to_models(&mut batch_models).unwrap();
    assert!(
        max_abs_difference(&batch_norms, &serial_norms) <= 1.0e-3,
        "vectorized norms {batch_norms:?} != serial norms {serial_norms:?}"
    );
    assert!(max_abs_difference(&batch_scales, &serial_scales) <= 1.0e-6);
    for (batch_model, serial_model) in batch_models.iter().zip(&serial_models) {
        assert!(max_abs_difference(&batch_model.weights.w1, &serial_model.weights.w1) <= 1.0e-6);
        assert!(max_abs_difference(&batch_model.weights.b1, &serial_model.weights.b1) <= 1.0e-6);
        assert!(max_abs_difference(&batch_model.weights.w2, &serial_model.weights.w2) <= 1.0e-6);
        assert!(max_abs_difference(&batch_model.weights.b2, &serial_model.weights.b2) <= 1.0e-6);
    }
}

#[test]
fn single_oracle_trajectory_batched_forward_matches_independent_base() {
    let device = BurnDevice::default();
    let config = NpaConfig::growing_2d();
    let model = NpaModel::upstream_seeded(config.clone(), 41);
    let trajectories = 3;
    let particles = 5;
    let feature_dims = config.perception_dims();
    let features = tensor3(
        (0..trajectories * particles * feature_dims)
            .map(|index| ((index % 37) as f32 - 18.0) * 0.003)
            .collect(),
        [trajectories, particles, feature_dims],
        &device,
    );
    let batch = BurnBaseBatch::from_models(std::slice::from_ref(&model), &device).unwrap();
    let actual = batch.forward(features.clone());
    let params = BurnBaseParams::from_model(&model, &device).unwrap();
    let rows = trajectories * particles;
    let flattened = features.reshape([rows, feature_dims]);
    let expected = relu(
        flattened.matmul(params.w1.transpose()) + params.b1.expand([rows, config.hidden_dims]),
    )
    .matmul(params.w2.transpose())
    .reshape([trajectories, particles, config.update_dims()]);
    assert!(
        max_abs_difference(
            &tensor3_vec(actual.inner()).unwrap(),
            &tensor3_vec(expected.inner()).unwrap(),
        ) <= 1.0e-6
    );
}

#[test]
fn oracle_batch_model_view_matches_each_independent_base() {
    let device = BurnDevice::default();
    let config = NpaConfig::growing_2d();
    let models = [
        NpaModel::upstream_seeded(config.clone(), 41),
        NpaModel::upstream_seeded(config.clone(), 73),
    ];
    let batch = BurnBaseBatch::from_models(&models, &device).unwrap();

    for (index, model) in models.iter().enumerate() {
        let selected = batch.model(index);
        assert!(max_abs_difference(&tensor_vec(selected.w1.inner()).unwrap(), &model.weights.w1) <= 1.0e-6);
        assert!(max_abs_difference(&tensor_vec(selected.b1.inner()).unwrap(), &model.weights.b1) <= 1.0e-6);
        assert!(max_abs_difference(&tensor_vec(selected.w2.inner()).unwrap(), &model.weights.w2) <= 1.0e-6);
        assert!(max_abs_difference(&tensor_vec(selected.b2.inner()).unwrap(), &model.weights.b2) <= 1.0e-6);
    }
}

#[test]
fn module_token_v3_burn_forward_matches_serialized_inference() {
    let device = BurnDevice::default();
    let config = NpaConfig::growing_2d();
    let rank = 2;
    let chunk_size = 16;
    let hidden_dims = 4;
    let attention_heads = 2;
    let embed_dims = 3;
    let token_count = 3;
    let layout =
        crate::hyper::adapter_layout::AdapterParameterLayout2d::new(&config, rank, chunk_size)
            .unwrap();
    let token_w = vec![0.7, 0.1, 0.0, 0.0, 0.8, 0.2, 0.1, 0.0, 0.9, 0.3, 0.4, 0.5];
    let token_b = vec![0.01, 0.02, 0.03, 0.04];
    let token_gate_w = layout.structured_query_initialization(hidden_dims, 0.2);
    let token_gate_b = layout.structured_query_initialization(hidden_dims, 0.1);
    let state_w = vec![0.0; hidden_dims * chunk_size];
    let time_w = vec![0.0; hidden_dims];
    let output_w = (0..chunk_size * hidden_dims)
        .map(|index| ((index % 11) as f32 - 5.0) * 0.003)
        .collect::<Vec<_>>();
    let output_b = vec![0.0; layout.padded_parameter_count()];
    let condition = vec![0.9, 0.1, 0.2, 0.1, 0.8, 0.3, 0.2, 0.3, 0.9];
    let output_dims = layout.parameter_count;

    let generator = BurnE2eGeneratorParams {
        kind: E2eHyperGeneratorKind::ModuleTokenDecoder,
        token_w: tracked_tensor(token_w.clone(), [hidden_dims, embed_dims], &device),
        token_b: tracked_tensor(token_b.clone(), [1, hidden_dims], &device),
        token_gate_w: tracked_tensor(
            token_gate_w.clone(),
            [layout.chunk_count, hidden_dims],
            &device,
        ),
        token_gate_b: tracked_tensor(
            token_gate_b.clone(),
            [layout.chunk_count, hidden_dims],
            &device,
        ),
        state_w: tracked_tensor(state_w.clone(), [hidden_dims, chunk_size], &device),
        time_w: tracked_tensor(time_w.clone(), [hidden_dims, 1], &device),
        output_w: tracked_tensor(output_w.clone(), [chunk_size, hidden_dims], &device),
        output_b: tracked_tensor(output_b.clone(), [layout.chunk_count, chunk_size], &device),
        condition_control_w: tracked_tensor(
            vec![0.0; config.update_dims() * hidden_dims],
            [config.update_dims(), hidden_dims],
            &device,
        ),
        condition_control_b: tracked_tensor(
            vec![0.0; config.update_dims()],
            [1, config.update_dims()],
            &device,
        ),
        condition_control_state_w: tracked_tensor(
            vec![0.0; hidden_dims * config.state_dims],
            [hidden_dims, config.state_dims],
            &device,
        ),
        hidden_dims,
        token_attention_heads: attention_heads,
        softmax_token_attention: true,
        canonical_full_rank_lora: false,
        adapter_constants: tracked_tensor(vec![0.0; output_dims], [1, output_dims], &device),
        adapter_trainable_mask: tracked_tensor(vec![1.0; output_dims], [1, output_dims], &device),
        adapter_parameter_segments: BurnE2eGeneratorParams::adapter_parameter_segments(
            &config, rank,
        ),
        output_dims,
        output_scale: 1.0,
        sample_steps: 1,
        adapter_chunk_size: chunk_size,
        output_chunks: layout.chunk_count,
        row_flow: None,
        amortization_residual_table: None,
        amortization_gradient_layout: None,
        amortization_learning_rate_scale: 1.0,
        amortization_grad_normalization: false,
    };
    let burn_vector = tensor_vec(
        generator
            .spatial_token_adapter_vector_batch(
                tensor3(condition.clone(), [1, token_count, embed_dims], &device),
                &config,
                rank,
            )
            .inner(),
    )
    .unwrap();
    let hyper = E2eHyperNpa2d {
        version: 1,
        architecture: E2eHyperGeneratorKind::ModuleTokenDecoder
            .artifact_architecture()
            .to_string(),
        backend: Some("test".to_string()),
        condition_encoder: Some("dino-vits-full-tokens".to_string()),
        condition_token_count: Some(token_count),
        condition_embed_dims: Some(embed_dims),
        condition_token_grid_width: None,
        condition_token_grid_height: None,
        condition_image_size: Some(224),
        condition_alpha_mode: Some("composite-white".to_string()),
        condition_rgb_channels: Some(false),
        condition_rgb_channel_scale: Some(1.0),
        condition_alpha_channel: Some(false),
        condition_alpha_channel_scale: Some(1.0),
        condition_patch_pixels: None,
        condition_l2_normalize_features: Some(false),
        condition_resize_mode: Some("stretch".to_string()),
        condition_application: Some("static-adapter".to_string()),
        shared_base_sha256: None,
        hidden_dims,
        token_attention_heads: attention_heads,
        attention_normalization: Some(crate::hyper::e2e::E2E_HYPER_ATTENTION_SOFTMAX.to_string()),
        output_dims,
        sample_steps: 1,
        output_scale: 1.0,
        adapter_rank: Some(rank),
        adapter_alpha: Some(rank as f32),
        adapter_parameterization: Some(E2E_HYPER_ADAPTER_FACTORIZED.to_string()),
        adapter_output_bias: None,
        adapter_chunk_size: Some(chunk_size),
        spatial_condition_control: None,
        spatial_condition_control_scale: None,
        spatial_condition_control_sigma: None,
        spatial_condition_state_control: None,
        row_flow: None,
        weights: E2eHyperNpa2dWeights {
            token_w,
            token_b,
            token_gate_w,
            token_gate_b,
            state_w,
            time_w,
            output_w,
            output_b,
            condition_control_w: Vec::new(),
            condition_control_b: Vec::new(),
            condition_control_state_w: Vec::new(),
            row_flow: Vec::new(),
        },
    };
    let inference_vector = hyper
        .predict_adapter(&config, &condition)
        .unwrap()
        .to_parameter_vector();

    assert!(
        max_abs_difference(&burn_vector, &inference_vector) < 2.0e-5,
        "Burn and serialized inference module-token v3 forwards diverged"
    );
}

#[test]
fn conditional_row_flow_burn_forward_matches_serialized_inference() {
    let config = NpaConfig::growing_2d();
    let layout = NpaParameterRowLayout2d::new(&config);
    let rank = layout.canonical_rank();
    let flow = ConditionalRowFlowConfig {
        layers: 1,
        width: 8,
        heads: 2,
        ffn_dims: 16,
        condition_dims: 3,
        condition_tokens: 2,
        row_count: layout.row_count(),
        max_row_dims: layout.max_row_dims(),
        row_value_dims: layout.rows().iter().map(|row| row.value_dims).collect(),
        sample_steps: 2,
        source_seed: 31,
        source_scale: 1.0,
        solver: crate::hyper::CONDITIONAL_ROW_FLOW_SOLVER_HEUN.to_string(),
        row_rms: vec![0.02; layout.row_count()],
    };
    let weights = ConditionalRowFlowWeights::seeded(&flow, 37).unwrap();
    let hyper = E2eHyperNpa2d {
        version: 2,
        architecture: E2E_HYPER_ARCH_CONDITIONAL_ROW_FLOW.to_string(),
        backend: Some("test".to_string()),
        condition_encoder: Some("dino-vits-full-tokens".to_string()),
        condition_token_count: Some(flow.condition_tokens),
        condition_embed_dims: Some(flow.condition_dims),
        condition_token_grid_width: Some(1),
        condition_token_grid_height: Some(1),
        condition_image_size: Some(224),
        condition_alpha_mode: Some("composite-white".to_string()),
        condition_rgb_channels: Some(false),
        condition_rgb_channel_scale: Some(1.0),
        condition_alpha_channel: Some(false),
        condition_alpha_channel_scale: Some(1.0),
        condition_patch_pixels: None,
        condition_l2_normalize_features: Some(false),
        condition_resize_mode: Some("stretch".to_string()),
        condition_application: Some("static-adapter".to_string()),
        shared_base_sha256: None,
        hidden_dims: flow.width,
        token_attention_heads: flow.heads,
        attention_normalization: Some(E2E_HYPER_ATTENTION_SOFTMAX.to_string()),
        output_dims: layout.parameter_count(),
        sample_steps: flow.sample_steps,
        output_scale: flow.source_scale,
        adapter_rank: Some(rank),
        adapter_alpha: Some(rank as f32),
        adapter_parameterization: Some(E2E_HYPER_ADAPTER_DENSE_ROW_RESIDUAL.to_string()),
        adapter_output_bias: None,
        adapter_chunk_size: None,
        spatial_condition_control: None,
        spatial_condition_control_scale: None,
        spatial_condition_control_sigma: None,
        spatial_condition_state_control: None,
        row_flow: Some(flow),
        weights: E2eHyperNpa2dWeights {
            token_w: Vec::new(),
            token_b: Vec::new(),
            token_gate_w: Vec::new(),
            token_gate_b: Vec::new(),
            state_w: Vec::new(),
            time_w: Vec::new(),
            output_w: Vec::new(),
            output_b: Vec::new(),
            condition_control_w: Vec::new(),
            condition_control_b: Vec::new(),
            condition_control_state_w: Vec::new(),
            row_flow: weights.values,
        },
    };
    let condition = vec![0.1, -0.2, 0.3, 0.4, 0.5, -0.6];
    let expected = hyper
        .predict_adapter(&config, &condition)
        .unwrap()
        .to_parameter_vector();
    let actual = predict_conditional_row_flow_adapter(&hyper, &config, &condition)
        .unwrap()
        .to_parameter_vector();
    assert!(
        max_abs_difference(&actual, &expected) <= 2.0e-5,
        "Burn and serialized conditional row-flow inference diverged"
    );

    let mut upstream_aligned = hyper;
    upstream_aligned.adapter_output_bias = Some(false);
    let expected = upstream_aligned
        .predict_adapter(&config, &condition)
        .unwrap();
    let actual =
        predict_conditional_row_flow_adapter(&upstream_aligned, &config, &condition).unwrap();
    assert!(expected.b2_delta.iter().all(|value| *value == 0.0));
    assert!(actual.b2_delta.iter().all(|value| *value == 0.0));
    assert!(
        max_abs_difference(
            &actual.to_parameter_vector(),
            &expected.to_parameter_vector(),
        ) <= 2.0e-5,
        "Burn and serialized no-output-bias row-flow inference diverged"
    );
}

#[test]
fn conditional_row_flow_self_rectification_has_reproducible_nonzero_gradients() {
    let npa = NpaConfig::growing_2d();
    let layout = NpaParameterRowLayout2d::new(&npa);
    let flow = ConditionalRowFlowConfig {
        layers: 1,
        width: 8,
        heads: 2,
        ffn_dims: 16,
        condition_dims: 3,
        condition_tokens: 2,
        row_count: layout.row_count(),
        max_row_dims: layout.max_row_dims(),
        row_value_dims: layout.rows().iter().map(|row| row.value_dims).collect(),
        sample_steps: 2,
        source_seed: 41,
        source_scale: 1.0,
        solver: crate::hyper::CONDITIONAL_ROW_FLOW_SOLVER_HEUN.to_string(),
        row_rms: vec![0.02; layout.row_count()],
    };
    let weights = ConditionalRowFlowWeights::seeded(&flow, 43).unwrap();
    let device = BurnDevice::default();
    let params = BurnRowFlowParams::from_values(flow, &weights.values, &npa, &device).unwrap();
    let condition = tensor3(vec![0.1, -0.2, 0.3, 0.4, 0.5, -0.6], [1, 2, 3], &device);
    let output_weight = params.tensors[params.tensors.len() - 2].clone();
    let (endpoint, prepared) = params.sample_rows_with_prepared(condition.clone(), &npa);
    let endpoint_dims = endpoint.shape().dims::<3>();
    let endpoint = (detach3(endpoint)
        + Tensor::<BurnBackend, 3>::ones(endpoint_dims, &device).mul_scalar(0.01))
    .require_grad();
    let endpoint_probe = endpoint.clone();
    let fresh_loss =
        params.self_rectification_loss_to_endpoint(condition, endpoint.clone(), &npa, 47);
    let loss = params.self_rectification_loss_to_endpoint_prepared(&prepared, endpoint, &npa, 47);
    let loss_value = loss.clone().inner().into_scalar();
    let fresh_loss_value = fresh_loss.inner().into_scalar();
    assert!((loss_value - fresh_loss_value).abs() < 1.0e-7);
    assert!(loss_value.is_finite() && loss_value > 0.0);
    let mut grads = loss.backward();
    assert!(
        endpoint_probe.grad_remove(&mut grads).is_none(),
        "rectified-flow endpoint targets must remain detached"
    );
    let gradient = tensor_vec(
        output_weight
            .grad_remove(&mut grads)
            .expect("self-rectification output gradient"),
    )
    .unwrap();
    assert!(gradient.iter().all(|value| value.is_finite()));
    assert!(gradient.iter().any(|value| value.abs() > 1.0e-10));
}

#[test]
fn conditional_row_flow_training_step_override_keeps_full_inference_sampler() {
    let npa = NpaConfig::growing_2d();
    let layout = NpaParameterRowLayout2d::new(&npa);
    let flow = ConditionalRowFlowConfig {
        layers: 1,
        width: 8,
        heads: 2,
        ffn_dims: 16,
        condition_dims: 3,
        condition_tokens: 2,
        row_count: layout.row_count(),
        max_row_dims: layout.max_row_dims(),
        row_value_dims: layout.rows().iter().map(|row| row.value_dims).collect(),
        sample_steps: 4,
        source_seed: 53,
        source_scale: 1.0,
        solver: crate::hyper::CONDITIONAL_ROW_FLOW_SOLVER_HEUN.to_string(),
        row_rms: vec![0.02; layout.row_count()],
    };
    let weights = ConditionalRowFlowWeights::seeded(&flow, 59).unwrap();
    let device = BurnDevice::default();
    let params = BurnRowFlowParams::from_values(flow, &weights.values, &npa, &device).unwrap();
    let condition = tensor3(vec![0.1, -0.2, 0.3, 0.4, 0.5, -0.6], [1, 2, 3], &device);
    let (inference, _) = params.sample_rows_with_prepared(condition.clone(), &npa);
    let (explicit_full, _) = params.sample_rows_with_prepared_steps(condition.clone(), &npa, 4);
    let (training, _) = params.sample_rows_with_prepared_steps(condition, &npa, 1);
    let inference = tensor3_vec(inference.inner()).unwrap();
    let explicit_full = tensor3_vec(explicit_full.inner()).unwrap();
    let training = tensor3_vec(training.inner()).unwrap();
    assert!(inference.iter().all(|value| value.is_finite()));
    assert!(training.iter().all(|value| value.is_finite()));
    assert!(max_abs_difference(&inference, &explicit_full) <= 1.0e-7);
    assert_eq!(inference.len(), training.len());
    assert_eq!(params.config.sample_steps, 4);
}

#[test]
fn conditional_row_flow_endpoint_loss_uses_normalized_dense_rows() {
    let npa = NpaConfig::growing_2d();
    let layout = NpaParameterRowLayout2d::new(&npa);
    let rank = layout.canonical_rank();
    let flow = ConditionalRowFlowConfig {
        layers: 1,
        width: 8,
        heads: 2,
        ffn_dims: 16,
        condition_dims: 3,
        condition_tokens: 2,
        row_count: layout.row_count(),
        max_row_dims: layout.max_row_dims(),
        row_value_dims: layout.rows().iter().map(|row| row.value_dims).collect(),
        sample_steps: 2,
        source_seed: 47,
        source_scale: 1.0,
        solver: crate::hyper::CONDITIONAL_ROW_FLOW_SOLVER_HEUN.to_string(),
        row_rms: vec![0.02; layout.row_count()],
    };
    let weights = ConditionalRowFlowWeights::seeded(&flow, 53).unwrap();
    let device = BurnDevice::default();
    let params = BurnRowFlowParams::from_values(flow, &weights.values, &npa, &device).unwrap();
    let sampled_rows = params.sample_rows(
        tensor3(vec![0.1, -0.2, 0.3, 0.4, 0.5, -0.6], [1, 2, 3], &device),
        &npa,
    );
    let teacher = BurnAdapterBatch::from_dense_residual_rows(detach3(sampled_rows.clone()), &npa)
        .to_parameter_vector();
    let zero_loss =
        params.endpoint_reconstruction_loss(sampled_rows.clone(), teacher, &npa, rank, rank as f32);
    let zero_loss_value = zero_loss.inner().into_scalar();
    assert!(
        zero_loss_value.abs() < 1.0e-6,
        "dense-row adapter roundtrip loss was {zero_loss_value:e}"
    );

    let zero_teacher_values =
        NpaLowRankAdapter::zeros(&npa, rank, rank as f32).to_parameter_vector();
    let zero_teacher_len = zero_teacher_values.len();
    let zero_teacher = tensor(zero_teacher_values, [1, zero_teacher_len], &device);
    let output_weight = params.tensors[params.tensors.len() - 2].clone();
    let loss =
        params.endpoint_reconstruction_loss(sampled_rows, zero_teacher, &npa, rank, rank as f32);
    let loss_value = loss.clone().inner().into_scalar();
    assert!(loss_value.is_finite() && loss_value > 0.0);
    let mut grads = loss.backward();
    let gradient = tensor_vec(
        output_weight
            .grad_remove(&mut grads)
            .expect("endpoint reconstruction output gradient"),
    )
    .unwrap();
    assert!(gradient.iter().all(|value| value.is_finite()));
    assert!(gradient.iter().any(|value| value.abs() > 1.0e-10));
}

#[test]
fn conditional_row_flow_amortization_distills_endpoint_without_teacher_gradient() {
    let npa = NpaConfig::growing_2d();
    let layout = NpaParameterRowLayout2d::new(&npa);
    let flow = ConditionalRowFlowConfig {
        layers: 1,
        width: 8,
        heads: 2,
        ffn_dims: 16,
        condition_dims: 3,
        condition_tokens: 2,
        row_count: layout.row_count(),
        max_row_dims: layout.max_row_dims(),
        row_value_dims: layout.rows().iter().map(|row| row.value_dims).collect(),
        sample_steps: 2,
        source_seed: 57,
        source_scale: 1.0,
        solver: crate::hyper::CONDITIONAL_ROW_FLOW_SOLVER_HEUN.to_string(),
        row_rms: vec![0.02; layout.row_count()],
    };
    let weights = ConditionalRowFlowWeights::seeded(&flow, 61).unwrap();
    let device = BurnDevice::default();
    let params = BurnRowFlowParams::from_values(flow, &weights.values, &npa, &device).unwrap();
    let generated = params.sample_rows(
        tensor3(vec![0.1, -0.2, 0.3, 0.4, 0.5, -0.6], [1, 2, 3], &device),
        &npa,
    );
    let zero =
        params.amortization_distillation_loss(generated.clone(), detach3(generated.clone()), &npa);
    assert!(zero.inner().into_scalar().abs() < 1.0e-7);

    let dims = generated.shape().dims::<3>();
    let teacher = (detach3(generated.clone())
        + Tensor::<BurnBackend, 3>::ones(dims, &device).mul_scalar(0.01))
    .require_grad();
    let teacher_probe = teacher.clone();
    let output_weight = params.tensors[params.tensors.len() - 2].clone();
    let loss = params.amortization_distillation_loss(generated, teacher, &npa);
    let loss_value = loss.clone().inner().into_scalar();
    assert!(loss_value.is_finite() && loss_value > 0.0);
    let mut grads = loss.backward();
    assert!(teacher_probe.grad_remove(&mut grads).is_none());
    let gradient = tensor_vec(
        output_weight
            .grad_remove(&mut grads)
            .expect("amortization distillation output gradient"),
    )
    .unwrap();
    assert!(gradient.iter().all(|value| value.is_finite()));
    assert!(gradient.iter().any(|value| value.abs() > 1.0e-10));
}

#[test]
fn conditional_row_flow_batched_rows_match_independent_inference() {
    let npa = NpaConfig::growing_2d();
    let layout = NpaParameterRowLayout2d::new(&npa);
    let flow = ConditionalRowFlowConfig {
        layers: 1,
        width: 8,
        heads: 2,
        ffn_dims: 16,
        condition_dims: 3,
        condition_tokens: 2,
        row_count: layout.row_count(),
        max_row_dims: layout.max_row_dims(),
        row_value_dims: layout.rows().iter().map(|row| row.value_dims).collect(),
        sample_steps: 2,
        source_seed: 53,
        source_scale: 1.0,
        solver: crate::hyper::CONDITIONAL_ROW_FLOW_SOLVER_HEUN.to_string(),
        row_rms: vec![0.02; layout.row_count()],
    };
    let weights = ConditionalRowFlowWeights::seeded(&flow, 59).unwrap();
    let device = BurnDevice::default();
    let params = BurnRowFlowParams::from_values(flow, &weights.values, &npa, &device).unwrap();
    let values = vec![
        0.1, -0.2, 0.3, 0.4, 0.5, -0.6, -0.7, 0.8, 0.9, 0.2, -0.3, 0.6,
    ];
    let batched = tensor3_vec(
        params
            .sample_rows(tensor3(values.clone(), [2, 2, 3], &device), &npa)
            .inner(),
    )
    .unwrap();
    let row_values = layout.row_count() * layout.max_row_dims();
    for batch in 0..2 {
        let condition_start = batch * 6;
        let independent = tensor3_vec(
            params
                .sample_rows(
                    tensor3(
                        values[condition_start..condition_start + 6].to_vec(),
                        [1, 2, 3],
                        &device,
                    ),
                    &npa,
                )
                .inner(),
        )
        .unwrap();
        assert!(
            max_abs_difference(
                &batched[batch * row_values..(batch + 1) * row_values],
                &independent,
            ) < 2.0e-5,
            "batched row-flow inference mixed per-image conditions"
        );
    }
}

#[test]
fn row_flow_endpoint_bridge_matches_direct_multi_chunk_vjp() {
    let npa = NpaConfig::growing_2d();
    let layout = NpaParameterRowLayout2d::new(&npa);
    let flow = ConditionalRowFlowConfig {
        layers: 1,
        width: 8,
        heads: 2,
        ffn_dims: 16,
        condition_dims: 3,
        condition_tokens: 2,
        row_count: layout.row_count(),
        max_row_dims: layout.max_row_dims(),
        row_value_dims: layout.rows().iter().map(|row| row.value_dims).collect(),
        sample_steps: 2,
        source_seed: 67,
        source_scale: 1.0,
        solver: crate::hyper::CONDITIONAL_ROW_FLOW_SOLVER_HEUN.to_string(),
        row_rms: vec![0.02; layout.row_count()],
    };
    let weights = ConditionalRowFlowWeights::seeded(&flow, 71).unwrap();
    let device = BurnDevice::default();
    let direct =
        BurnRowFlowParams::from_values(flow.clone(), &weights.values, &npa, &device).unwrap();
    let bridged = BurnRowFlowParams::from_values(flow, &weights.values, &npa, &device).unwrap();
    let condition_values = vec![
        0.1, -0.2, 0.3, 0.4, 0.5, -0.6, -0.3, 0.2, 0.7, 0.6, -0.4, 0.1,
    ];
    let expansion = [0, 0, 1, 1];
    let output_dims = [expansion.len(), layout.row_count(), layout.max_row_dims()];
    let elements = output_dims.iter().product::<usize>();
    let coefficient_a = tensor3(
        (0..elements)
            .map(|index| ((index as f32 + 1.0) * 0.017).sin() * 0.01)
            .collect(),
        output_dims,
        &device,
    );
    let coefficient_b = tensor3(
        (0..elements)
            .map(|index| ((index as f32 + 1.0) * 0.013).cos() * 0.02)
            .collect(),
        output_dims,
        &device,
    );

    let direct_rows =
        direct.sample_rows(tensor3(condition_values.clone(), [2, 2, 3], &device), &npa);
    let direct_rows = BurnAdapterBatch::from_dense_residual_rows(direct_rows, &npa)
        .select_rows(&expansion)
        .dense_residual_rows(&npa);
    let direct_objective = (direct_rows.clone().mul(coefficient_a.clone()).sum()
        + direct_rows.mul(coefficient_b.clone()).sum())
    .mul_scalar(0.5);
    let mut direct_grads = direct_objective.backward();
    let direct_gradients = direct
        .tensors
        .iter()
        .map(|parameter| {
            parameter
                .grad_remove(&mut direct_grads)
                .expect("direct row-flow gradient")
        })
        .collect::<Vec<_>>();

    let generated_rows = bridged.sample_rows(tensor3(condition_values, [2, 2, 3], &device), &npa);
    let mut bridge = BurnRowFlowEndpointBridge::new(generated_rows);
    for coefficient in [coefficient_a, coefficient_b] {
        let rows = bridge
            .adapter_batch(&npa, Some(&expansion))
            .dense_residual_rows(&npa);
        let mut chunk_grads = rows.mul(coefficient).sum().backward();
        bridge.accumulate(&mut chunk_grads);
    }
    let mut bridged_grads = bridge
        .objective(0.5)
        .expect("two chunks produce an endpoint objective")
        .backward();
    let bridged_gradients = bridged
        .tensors
        .iter()
        .map(|parameter| {
            parameter
                .grad_remove(&mut bridged_grads)
                .expect("bridged row-flow gradient")
        })
        .collect::<Vec<_>>();

    assert_eq!(direct_gradients.len(), bridged_gradients.len());
    for (tensor_index, (direct, bridged)) in direct_gradients
        .into_iter()
        .zip(bridged_gradients)
        .enumerate()
    {
        let direct = tensor_vec(direct).unwrap();
        let bridged = tensor_vec(bridged).unwrap();
        assert_eq!(direct.len(), bridged.len());
        let max_error = direct
            .iter()
            .zip(&bridged)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        let max_reference = direct
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_error <= 2.0e-5 + max_reference * 2.0e-4,
            "row-flow tensor {tensor_index} endpoint VJP mismatch: error={max_error:e} reference={max_reference:e}"
        );
    }

    let condition_values = vec![
        0.1, -0.2, 0.3, 0.4, 0.5, -0.6, -0.3, 0.2, 0.7, 0.6, -0.4, 0.1,
    ];
    let auxiliary_coefficient = tensor3(
        (0..elements)
            .map(|index| ((index as f32 + 1.0) * 0.029).sin() * 0.015)
            .collect(),
        output_dims,
        &device,
    );
    let (direct_rows, direct_condition) = direct
        .sample_rows_with_prepared(tensor3(condition_values.clone(), [2, 2, 3], &device), &npa);
    let direct_expanded = BurnAdapterBatch::from_dense_residual_rows(direct_rows.clone(), &npa)
        .select_rows(&expansion)
        .dense_residual_rows(&npa);
    let direct_objective = direct_expanded
        .mul(auxiliary_coefficient.clone())
        .sum()
        .mul_scalar(0.25)
        + direct
            .self_rectification_loss_to_endpoint_prepared(&direct_condition, direct_rows, &npa, 79)
            .mul_scalar(0.7);
    let mut direct_grads = direct_objective.backward();
    let direct_gradients = direct
        .tensors
        .iter()
        .map(|parameter| {
            parameter
                .grad_remove(&mut direct_grads)
                .expect("direct row-flow auxiliary gradient")
        })
        .collect::<Vec<_>>();

    let (bridged_rows, bridged_condition) =
        bridged.sample_rows_with_prepared(tensor3(condition_values, [2, 2, 3], &device), &npa);
    let mut bridge =
        BurnRowFlowEndpointBridge::with_prepared_condition(bridged_rows, bridged_condition);
    let expanded_rows = bridge
        .adapter_batch(&npa, Some(&expansion))
        .dense_residual_rows(&npa);
    let mut endpoint_grads = expanded_rows.mul(auxiliary_coefficient).sum().backward();
    bridge.accumulate(&mut endpoint_grads);
    let bridged_objective = bridge
        .objective(0.25)
        .expect("endpoint gradient creates a bridge objective")
        + bridged
            .self_rectification_loss_to_endpoint_prepared(
                &bridge
                    .prepared_condition()
                    .expect("bridge retains the prepared condition"),
                bridge.generated_rows(),
                &npa,
                79,
            )
            .mul_scalar(0.7);
    let mut bridged_grads = bridged_objective.backward();
    let bridged_gradients = bridged
        .tensors
        .iter()
        .map(|parameter| {
            parameter
                .grad_remove(&mut bridged_grads)
                .expect("bridged row-flow auxiliary gradient")
        })
        .collect::<Vec<_>>();

    for (tensor_index, (direct, bridged)) in direct_gradients
        .into_iter()
        .zip(bridged_gradients)
        .enumerate()
    {
        let direct = tensor_vec(direct).unwrap();
        let bridged = tensor_vec(bridged).unwrap();
        let max_error = direct
            .iter()
            .zip(&bridged)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        let max_reference = direct
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_error <= 2.0e-5 + max_reference * 2.0e-4,
            "row-flow tensor {tensor_index} reused auxiliary gradient mismatch: error={max_error:e} reference={max_reference:e}"
        );
    }
}

#[test]
fn mixed_endpoint_bridge_matches_flow_and_endpoint_vjp() {
    let npa = NpaConfig::growing_2d();
    let layout = NpaParameterRowLayout2d::new(&npa);
    let flow = ConditionalRowFlowConfig {
        layers: 1,
        width: 8,
        heads: 2,
        ffn_dims: 16,
        condition_dims: 3,
        condition_tokens: 2,
        row_count: layout.row_count(),
        max_row_dims: layout.max_row_dims(),
        row_value_dims: layout.rows().iter().map(|row| row.value_dims).collect(),
        sample_steps: 2,
        source_seed: 83,
        source_scale: 1.0,
        solver: crate::hyper::CONDITIONAL_ROW_FLOW_SOLVER_HEUN.to_string(),
        row_rms: vec![0.02; layout.row_count()],
    };
    let weights = ConditionalRowFlowWeights::seeded(&flow, 89).unwrap();
    let device = BurnDevice::default();
    let direct =
        BurnRowFlowParams::from_values(flow.clone(), &weights.values, &npa, &device).unwrap();
    let bridged = BurnRowFlowParams::from_values(flow, &weights.values, &npa, &device).unwrap();
    let condition_values = vec![
        0.1, -0.2, 0.3, 0.4, 0.5, -0.6, -0.3, 0.2, 0.7, 0.6, -0.4, 0.1,
    ];
    let expansion = [0, 0, 1, 1];
    let row_dims = [expansion.len(), layout.row_count(), layout.max_row_dims()];
    let elements = row_dims.iter().product::<usize>();
    let endpoint_values = (0..elements)
        .map(|index| ((index as f32 + 1.0) * 0.011).sin() * 0.02)
        .collect::<Vec<_>>();
    let coefficient = tensor3(
        (0..elements)
            .map(|index| ((index as f32 + 1.0) * 0.019).cos() * 0.01)
            .collect(),
        row_dims,
        &device,
    );
    let mix = Tensor::<BurnBackend, 3>::from_data(
        TensorData::new(vec![0.0, 0.75, 0.0, 0.75], [4, 1, 1]),
        &device,
    )
    .expand(row_dims);

    let direct_generated =
        direct.sample_rows(tensor3(condition_values.clone(), [2, 2, 3], &device), &npa);
    let direct_generated = BurnAdapterBatch::from_dense_residual_rows(direct_generated, &npa)
        .select_rows(&expansion)
        .dense_residual_rows(&npa);
    let direct_endpoint = tensor3(endpoint_values.clone(), row_dims, &device).require_grad();
    let direct_endpoint_probe = direct_endpoint.clone();
    let direct_rows =
        direct_generated.mul(mix.clone().neg().add_scalar(1.0)) + direct_endpoint.mul(mix.clone());
    let direct_rows =
        BurnAdapterBatch::from_dense_residual_rows(direct_rows, &npa).dense_residual_rows(&npa);
    let mut direct_grads = direct_rows.mul(coefficient.clone()).sum().backward();
    let direct_flow_gradients = direct
        .tensors
        .iter()
        .map(|parameter| {
            parameter
                .grad_remove(&mut direct_grads)
                .expect("direct mixed row-flow gradient")
        })
        .collect::<Vec<_>>();
    let direct_endpoint_gradient = direct_endpoint_probe
        .grad_remove(&mut direct_grads)
        .expect("direct mixed endpoint gradient");

    let (bridged_generated, prepared) =
        bridged.sample_rows_with_prepared(tensor3(condition_values, [2, 2, 3], &device), &npa);
    let bridged_expanded =
        BurnAdapterBatch::from_dense_residual_rows(bridged_generated.clone(), &npa)
            .select_rows(&expansion)
            .dense_residual_rows(&npa);
    let bridged_endpoint = tensor3(endpoint_values, row_dims, &device).require_grad();
    let bridged_endpoint_probe = bridged_endpoint.clone();
    let mixed_rows =
        bridged_expanded.mul(mix.clone().neg().add_scalar(1.0)) + bridged_endpoint.mul(mix);
    let mut bridge =
        BurnRowFlowEndpointBridge::with_mixed_endpoint(bridged_generated, mixed_rows, prepared);
    let bridged_rows = bridge
        .adapter_batch(&npa, Some(&expansion))
        .dense_residual_rows(&npa);
    assert_eq!(bridged_rows.shape().dims::<3>(), row_dims);
    let mut rollout_grads = bridged_rows.mul(coefficient).sum().backward();
    bridge.accumulate(&mut rollout_grads);
    let mut bridged_grads = bridge
        .objective(1.0)
        .expect("mixed endpoint bridge accumulated a VJP")
        .backward();
    let bridged_flow_gradients = bridged
        .tensors
        .iter()
        .map(|parameter| {
            parameter
                .grad_remove(&mut bridged_grads)
                .expect("bridged mixed row-flow gradient")
        })
        .collect::<Vec<_>>();
    let bridged_endpoint_gradient = bridged_endpoint_probe
        .grad_remove(&mut bridged_grads)
        .expect("bridged mixed endpoint gradient");

    for (tensor_index, (direct, bridged)) in direct_flow_gradients
        .into_iter()
        .zip(bridged_flow_gradients)
        .enumerate()
    {
        let direct = tensor_vec(direct).unwrap();
        let bridged = tensor_vec(bridged).unwrap();
        let error = max_abs_difference(&direct, &bridged);
        let reference = direct
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        assert!(
            error <= 2.0e-5 + reference * 2.0e-4,
            "mixed row-flow tensor {tensor_index} VJP mismatch: error={error:e} reference={reference:e}"
        );
    }
    let direct_endpoint_gradient = tensor3_vec(direct_endpoint_gradient).unwrap();
    let bridged_endpoint_gradient = tensor3_vec(bridged_endpoint_gradient).unwrap();
    assert!(
        max_abs_difference(&direct_endpoint_gradient, &bridged_endpoint_gradient) <= 2.0e-6,
        "mixed endpoint tensor VJP mismatch"
    );
    assert!(
        bridged_endpoint_gradient
            .iter()
            .any(|value| value.abs() > 1.0e-8),
        "mixed endpoint bridge did not deliver endpoint gradients"
    );
}

#[test]
fn modulated_layer_norm_cube_matches_reference_value_and_gradients() {
    let device = BurnDevice::default();
    let [batches, rows, dims] = [2, 3, 8];
    let input_values = (0..batches * rows * dims)
        .map(|index| ((index as f32 * 0.37).sin() + index as f32 * 0.003) * 0.4)
        .collect::<Vec<_>>();
    let shift_values = (0..batches * dims)
        .map(|index| (index as f32 * 0.19).cos() * 0.07)
        .collect::<Vec<_>>();
    let scale_values = (0..batches * dims)
        .map(|index| (index as f32 * 0.23).sin() * 0.11)
        .collect::<Vec<_>>();
    let output_weights = tensor3(
        (0..batches * rows * dims)
            .map(|index| ((index as f32 + 1.0) * 0.13).cos() * 0.3)
            .collect(),
        [batches, rows, dims],
        &device,
    );

    let custom_input = tensor3(input_values.clone(), [batches, rows, dims], &device).require_grad();
    let custom_shift = tensor(shift_values.clone(), [batches, dims], &device).require_grad();
    let custom_scale = tensor(scale_values.clone(), [batches, dims], &device).require_grad();
    let custom = modulated_layer_norm3(
        custom_input.clone(),
        custom_shift.clone(),
        custom_scale.clone(),
    );
    let custom_values = tensor3_vec(custom.clone().inner()).unwrap();
    let mut custom_grads = custom.mul(output_weights.clone()).sum().backward();
    let custom_input_grad = tensor3_vec(
        custom_input
            .grad_remove(&mut custom_grads)
            .expect("custom input gradient"),
    )
    .unwrap();
    let custom_shift_grad = tensor_vec(
        custom_shift
            .grad_remove(&mut custom_grads)
            .expect("custom shift gradient"),
    )
    .unwrap();
    let custom_scale_grad = tensor_vec(
        custom_scale
            .grad_remove(&mut custom_grads)
            .expect("custom scale gradient"),
    )
    .unwrap();

    let reference_input = tensor3(input_values, [batches, rows, dims], &device).require_grad();
    let reference_shift = tensor(shift_values, [batches, dims], &device).require_grad();
    let reference_scale = tensor(scale_values, [batches, dims], &device).require_grad();
    let reference = super::row_flow::layer_norm3(reference_input.clone()).mul(
        reference_scale
            .clone()
            .add_scalar(1.0)
            .unsqueeze_dim::<3>(1)
            .expand([batches, rows, dims]),
    ) + reference_shift
        .clone()
        .unsqueeze_dim::<3>(1)
        .expand([batches, rows, dims]);
    let reference_values = tensor3_vec(reference.clone().inner()).unwrap();
    let mut reference_grads = reference.mul(output_weights).sum().backward();
    let reference_input_grad = tensor3_vec(
        reference_input
            .grad_remove(&mut reference_grads)
            .expect("reference input gradient"),
    )
    .unwrap();
    let reference_shift_grad = tensor_vec(
        reference_shift
            .grad_remove(&mut reference_grads)
            .expect("reference shift gradient"),
    )
    .unwrap();
    let reference_scale_grad = tensor_vec(
        reference_scale
            .grad_remove(&mut reference_grads)
            .expect("reference scale gradient"),
    )
    .unwrap();

    assert!(
        max_abs_difference(&custom_values, &reference_values) <= 2.0e-4,
        "modulated layer norm value mismatch: max_abs_diff={}",
        max_abs_difference(&custom_values, &reference_values),
    );
    assert!(
        max_abs_difference(&custom_input_grad, &reference_input_grad) <= 5.0e-4,
        "modulated layer norm input VJP mismatch: max_abs_diff={}",
        max_abs_difference(&custom_input_grad, &reference_input_grad),
    );
    assert!(
        max_abs_difference(&custom_shift_grad, &reference_shift_grad) <= 5.0e-4,
        "modulated layer norm shift VJP mismatch: max_abs_diff={}",
        max_abs_difference(&custom_shift_grad, &reference_shift_grad),
    );
    assert!(
        max_abs_difference(&custom_scale_grad, &reference_scale_grad) <= 5.0e-4,
        "modulated layer norm scale VJP mismatch: max_abs_diff={}",
        max_abs_difference(&custom_scale_grad, &reference_scale_grad),
    );
}

#[test]
fn dense_row_adapter_diagnostics_recover_generated_controller_values() {
    let npa = NpaConfig::growing_2d();
    let layout = NpaParameterRowLayout2d::new(&npa);
    let batches = 2;
    let p = npa.perception_dims();
    let h = npa.hidden_dims;
    let u = npa.update_dims();
    let mut rows = vec![777.0_f32; batches * layout.row_count() * layout.max_row_dims()];
    let mut expected = Vec::with_capacity(batches * layout.parameter_count());
    for batch in 0..batches {
        let row_offset = batch * layout.row_count() * layout.max_row_dims();
        for row in 0..h {
            for column in 0..p {
                let value = 1.0e-6 * (1 + batch * 10_000 + row * 100 + column) as f32;
                rows[row_offset + row * layout.max_row_dims() + column] = value;
                expected.push(value);
            }
        }
        for row in 0..h {
            let value = -1.0e-5 * (1 + batch * 100 + row) as f32;
            rows[row_offset + row * layout.max_row_dims() + p] = value;
            expected.push(value);
        }
        for row in 0..u {
            for column in 0..h {
                let value = 2.0e-6 * (1 + batch * 1_000 + row * 100 + column) as f32;
                rows[row_offset + (h + row) * layout.max_row_dims() + column] = value;
                expected.push(value);
            }
        }
        for row in 0..u {
            let value = -2.0e-5 * (1 + batch * 100 + row) as f32;
            rows[row_offset + (h + row) * layout.max_row_dims() + h] = value;
            expected.push(value);
        }
    }

    let device = BurnDevice::default();
    let adapter = BurnAdapterBatch::from_dense_residual_rows(
        tensor3(
            rows,
            [batches, layout.row_count(), layout.max_row_dims()],
            &device,
        ),
        &npa,
    );
    let actual = tensor_vec(adapter.dense_residual_vector(&npa).inner()).unwrap();
    assert_eq!(actual.len(), batches * layout.parameter_count());
    let (index, difference, actual_value, expected_value) =
        max_abs_difference_with_index(&actual, &expected);
    assert!(
        difference <= 2.0e-5,
        "dense diagnostics must inspect generated controller values, not canonical transport factors; max difference {difference} at {index}: actual={actual_value}, expected={expected_value}"
    );
    assert!(!actual.contains(&777.0));
}

fn max_abs_difference(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max)
}

fn test_tensor_values<const D: usize>(tensor: Tensor<InnerBackend, D>) -> Vec<f32> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .expect("test tensor should contain f32 values")
}

#[test]
fn tiled_attention_adjoint_matches_reference_value_and_gradients() {
    let device = BurnDevice::default();
    let shape_q = [2, 2, 3, 4];
    let shape_kv = [2, 2, 5, 4];
    let query_values = (0..shape_q.iter().product())
        .map(|index| (index as f32 * 0.17).sin() * 0.3)
        .collect::<Vec<_>>();
    let key_values = (0..shape_kv.iter().product())
        .map(|index| (index as f32 * 0.11).cos() * 0.25)
        .collect::<Vec<_>>();
    let value_values = (0..shape_kv.iter().product())
        .map(|index| (index as f32 * 0.07).sin() * 0.4)
        .collect::<Vec<_>>();
    let output_weights = (0..shape_q.iter().product())
        .map(|index| (index as f32 * 0.13).cos())
        .collect::<Vec<_>>();
    let tensor4 = |values: Vec<f32>, shape: [usize; 4]| {
        Tensor::<BurnBackend, 4>::from_data(TensorData::new(values, shape), &device).require_grad()
    };

    let tiled_query = tensor4(query_values.clone(), shape_q);
    let tiled_key = tensor4(key_values.clone(), shape_kv);
    let tiled_value = tensor4(value_values.clone(), shape_kv);
    let tiled =
        tiled_attention_adjoint(tiled_query.clone(), tiled_key.clone(), tiled_value.clone());

    let reference_query = tensor4(query_values, shape_q);
    let reference_key = tensor4(key_values, shape_kv);
    let reference_value = tensor4(value_values, shape_kv);
    let reference = softmax(
        reference_query
            .clone()
            .matmul(reference_key.clone().swap_dims(2, 3))
            .mul_scalar(0.5),
        3,
    )
    .matmul(reference_value.clone());
    let output_difference = max_abs_difference(
        &test_tensor_values(tiled.clone().inner()),
        &test_tensor_values(reference.clone().inner()),
    );
    assert!(
        output_difference < 2.0e-4,
        "tiled attention output difference {output_difference}"
    );

    let tiled_loss = tiled
        .mul(Tensor::<BurnBackend, 4>::from_data(
            TensorData::new(output_weights.clone(), shape_q),
            &device,
        ))
        .sum();
    let reference_loss = reference
        .mul(Tensor::<BurnBackend, 4>::from_data(
            TensorData::new(output_weights, shape_q),
            &device,
        ))
        .sum();
    let mut tiled_grads = tiled_loss.backward();
    let mut reference_grads = reference_loss.backward();
    for (label, tiled_tensor, reference_tensor) in [
        ("query", tiled_query, reference_query),
        ("key", tiled_key, reference_key),
        ("value", tiled_value, reference_value),
    ] {
        let tiled_gradient = test_tensor_values(
            tiled_tensor
                .grad_remove(&mut tiled_grads)
                .expect("tiled attention gradient"),
        );
        let reference_gradient = test_tensor_values(
            reference_tensor
                .grad_remove(&mut reference_grads)
                .expect("reference attention gradient"),
        );
        let difference = max_abs_difference(&tiled_gradient, &reference_gradient);
        assert!(
            difference < 5.0e-4,
            "tiled attention {label} gradient difference {difference}"
        );
    }
}

fn max_abs_difference_with_index(left: &[f32], right: &[f32]) -> (usize, f32, f32, f32) {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .enumerate()
        .map(|(idx, (left, right))| (idx, (left - right).abs(), *left, *right))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, 0.0, 0.0, 0.0))
}

fn test_burn_target(device: &BurnDevice, update_prob: f32, seed_scale: f32) -> BurnTargetExample {
    let target_cpu = crate::TargetImage2d {
        source_width: 1,
        source_height: 1,
        positions: vec![[0.0, 0.0]],
        colors: vec![[1.0, 1.0, 1.0]],
        pixel_size: 2.0,
        threshold: 0.05,
        aabb: [-1.0, 1.0, -1.0, 1.0],
    };
    BurnTargetExample {
        target_rgb: Tensor::<BurnBackend, 2>::zeros([1, 3], device),
        target_density: Tensor::<BurnBackend, 2>::zeros([1, 1], device),
        target_foreground: Tensor::<BurnBackend, 2>::zeros([1, 1], device),
        target_foreground_scale: 1.0,
        target_mean: Tensor::<BurnBackend, 2>::zeros([1, 2], device),
        target_positions: Tensor::<BurnBackend, 2>::zeros([1, 2], device),
        pixel_xy: Tensor::<BurnBackend, 2>::zeros([1, 2], device),
        pixel_size: 2.0,
        target_points: 1,
        particle_count: 4,
        update_prob,
        seed_scale,
        target_cpu,
    }
}

fn test_direct_config(particle_count: usize) -> DirectBasisTrainConfig {
    let npa_config = NpaConfig::growing_2d();
    let grid = burn_automata_kernels::HashGridConfig::growing_2d();
    DirectBasisTrainConfig {
        steps: 0,
        report_interval: 1,
        example_batch_size: 2,
        tbptt_chunk_steps: 1,
        loss_on_final_chunk_only: false,
        use_particle_pool: false,
        pool_size: 0,
        inject_seed_interval: 0,
        brush_size: 0.0,
        stopgrad_pos: npa_config.stopgrad_pos,
        stopgrad_state: npa_config.stopgrad_state,
        rollout_particles: particle_count,
        rollout_step_min: 1,
        rollout_steps: 1,
        update_prob: 1.0,
        seed: 13,
        seed_scale: 0.1,
        seed_mode: crate::ParticleSeed::UniformCircle,
        grid_eps: grid.eps,
        motion_scale: npa_config.alpha * npa_config.motion_eps(grid.eps),
        loss_config: crate::Target2dLossConfig::default(),
        target2d_loss_backend: Target2dLossBackend::Dense,
        perception_backend: PerceptionRolloutBackend::Dense,
        per_parameter_grad_normalization: false,
        base_sgd: SgdConfig {
            learning_rate: 0.0,
            weight_decay: 0.0,
            grad_clip_norm: 0.0,
        },
        adapter_sgd: SgdConfig {
            learning_rate: 0.0,
            weight_decay: 0.0,
            grad_clip_norm: 0.0,
        },
        adapter_l2_weight: 0.0,
        update_base: true,
        eval_examples: 1,
        eval_interval: 0,
        eval_batch_size: 1,
        eval_seed: 13,
        system_memory_budget_gb: None,
        gpu_memory_budget_gb: None,
        max_dense_train_particles: particle_count,
        max_dense_chunk_floats: 1_000_000,
        max_splat_chunk_floats: 1_000_000,
    }
}

#[test]
fn adaptive_deployment_rgb_metric_ignores_training_centering() {
    let device = BurnDevice::default();
    let target = test_burn_target(&device, 1.0, 0.1);
    let x = tensor3(
        vec![0.4, 0.3, 0.45, 0.35, 0.5, 0.4, 0.55, 0.45],
        [1, 4, 2],
        &device,
    );
    let s = tensor3(vec![0.0; 4 * 16], [1, 4, 16], &device);
    let particle_pixel_size = tensor(vec![2.0; 4], [1, 4], &device);
    let particle_output_scale = tensor(vec![1.0; 4], [1, 4], &device);
    let mut uncentered = test_direct_config(4);
    uncentered.loss_config.image_size = 1;
    uncentered.loss_config.center = false;
    let mut centered_training = uncentered;
    centered_training.loss_config.center = true;

    let metric = |config| {
        tensor1_vec(
            adaptive_uncentered_render_rgb_mse_batch(
                &x,
                &s,
                std::slice::from_ref(&target),
                &[0],
                config,
                particle_pixel_size.clone(),
                particle_output_scale.clone(),
            )
            .inner(),
        )
        .unwrap()[0]
    };
    assert_eq!(metric(uncentered), metric(centered_training));
}

#[test]
fn adaptive_variable_footprint_render_matches_host_contract() {
    let device = BurnDevice::default();
    let particle_count = 3;
    let state_dims = 4;
    let mut config = test_direct_config(particle_count);
    config.loss_config.image_size = 16;
    config.max_splat_chunk_floats = 1_000_000;
    let pixels = config.loss_config.image_size * config.loss_config.image_size;
    let target_cpu = crate::TargetImage2d {
        source_width: 1,
        source_height: 1,
        positions: vec![[0.0, 0.0]],
        colors: vec![[1.0, 1.0, 1.0]],
        pixel_size: 0.1,
        threshold: 0.05,
        aabb: [-1.0, 1.0, -1.0, 1.0],
    };
    let target = BurnTargetExample {
        target_rgb: Tensor::<BurnBackend, 2>::zeros([pixels, 3], &device),
        target_density: Tensor::<BurnBackend, 2>::zeros([pixels, 1], &device),
        target_foreground: Tensor::<BurnBackend, 2>::zeros([pixels, 1], &device),
        target_foreground_scale: 1.0,
        target_mean: Tensor::<BurnBackend, 2>::zeros([1, 2], &device),
        target_positions: Tensor::<BurnBackend, 2>::zeros([1, 2], &device),
        pixel_xy: tensor(
            pixel_xy_values(config.loss_config.image_size),
            [pixels, 2],
            &device,
        ),
        pixel_size: target_cpu.pixel_size,
        target_points: 4,
        particle_count,
        update_prob: 1.0,
        seed_scale: 0.1,
        target_cpu,
    };
    let positions = vec![-0.35, -0.2, 0.1, 0.25, 0.45, -0.3];
    let states = vec![
        0.0, -0.25, 0.1, 0.35, 0.0, 0.4, -0.2, 0.15, 0.0, 0.2, 0.3, -0.1,
    ];
    let pixel_sizes = vec![0.08, 0.16, 0.12];
    let output_scales = vec![1.25, 0.75, 1.5];
    let x = tensor3(positions.clone(), [1, particle_count, 2], &device);
    let s = tensor3(
        states.clone(),
        [1, particle_count, state_dims],
        &device,
    );
    let colors = s
        .narrow(2, state_dims - 3, 3)
        .add_scalar(0.5);
    let (burn_rgb, burn_density) = adaptive_splat_render_batch(
        &x,
        &colors,
        std::slice::from_ref(&target),
        &[0],
        config,
        tensor(pixel_sizes.clone(), [1, particle_count], &device),
        tensor(output_scales.clone(), [1, particle_count], &device),
    );
    let host_positions = positions
        .chunks_exact(2)
        .map(|position| [position[0], position[1], 0.0, 0.0])
        .collect::<Vec<_>>();
    let host = crate::target2d::render_variable_rollout_2d_splat(
        &host_positions,
        &states,
        state_dims,
        &pixel_sizes,
        &output_scales,
        config.loss_config,
        None,
    )
    .unwrap();
    let rgb_difference =
        max_abs_difference(&tensor3_vec(burn_rgb.inner()).unwrap(), &host.rgb);
    let density_difference =
        max_abs_difference(&tensor3_vec(burn_density.inner()).unwrap(), &host.density);
    assert!(
        rgb_difference < 2.0e-5 && density_difference < 2.0e-5,
        "adaptive Burn/host render mismatch: rgb={rgb_difference:e} density={density_difference:e}"
    );
}

#[test]
fn bounded_eval_rollout_matches_canonical_core_mask_stream() {
    let device = BurnDevice::default();
    let npa = NpaConfig::growing_2d();
    let grid = burn_automata_kernels::HashGridConfig::growing_2d();
    let model = NpaModel::upstream_seeded(npa.clone(), 71);
    let params = BurnBaseParams::from_model(&model, &device).unwrap();
    let adapter = BurnAdapterParams::from_adapter(
        &NpaLowRankAdapter::zeros(&npa, 1, 1.0),
        &model,
        &device,
    )
    .unwrap();
    let particle_count = 4;
    let steps = 3;
    let seed = 47;
    let seed_scale = 0.1;
    let mut config = test_direct_config(particle_count);
    config.rollout_steps = steps;
    config.tbptt_chunk_steps = steps;
    config.update_prob = 0.5;
    let target = test_burn_target(&device, config.update_prob, seed_scale);
    let (x, s) = seed_tensors(particle_count, config, seed_scale, seed, &device);
    let (x, s, _) = rollout_single_chunk(
        &params,
        &adapter,
        &target,
        x,
        s,
        config,
        &mut canonical_eval_mask_rng(seed),
        steps,
        Tensor::<BurnBackend, 1>::zeros([1], &device),
    );

    let reference = crate::run_rollout(
        &model,
        &grid,
        &crate::RolloutConfig {
            batch_size: 1,
            particle_count,
            steps,
            update_prob: config.update_prob,
            seed,
            seed_scale,
            ..crate::RolloutConfig::default()
        },
        config.seed_mode,
    )
    .unwrap();
    let expected_x = reference
        .positions
        .iter()
        .flat_map(|position| [position[0], position[1]])
        .collect::<Vec<_>>();
    let x_diff = max_abs_difference(&tensor_vec(x.inner()).unwrap(), &expected_x);
    let s_diff = max_abs_difference(&tensor_vec(s.inner()).unwrap(), &reference.states);
    assert!(
        x_diff < 2.0e-4 && s_diff < 2.0e-3,
        "bounded Burn eval diverged from canonical core rollout: x={x_diff:e} s={s_diff:e}"
    );
}

#[test]
fn adaptive_eval_rollout_reuses_the_seeded_mask_stream_for_one_model() {
    let device = BurnDevice::default();
    let model = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 71);
    let params =
        BurnBaseBatch::from_models(std::slice::from_ref(&model), &device).unwrap().detached();
    let particle_count = 4;
    let steps = 3;
    let seed = 47;
    let seed_scale = 0.1;
    let mut config = test_direct_config(particle_count);
    config.rollout_steps = steps;
    config.tbptt_chunk_steps = steps;
    config.update_prob = 0.5;
    let target = test_burn_target(&device, config.update_prob, seed_scale);
    let (x, s) = seed_tensors(particle_count, config, seed_scale, seed, &device);
    let x = x.reshape([1, particle_count, 2]);
    let s = s.reshape([1, particle_count, model.config.state_dims]);
    let material = Tensor::<BurnBackend, 2>::ones([1, particle_count], &device);
    let bandwidth =
        Tensor::<BurnBackend, 2>::ones([1, particle_count], &device).mul_scalar(config.grid_eps);
    let perception = burn_automata_kernels::AdaptivePerceptionConfig {
        graph_policy: burn_automata_kernels::AdaptiveGraphPolicy::RawSupport,
        max_neighbors: particle_count,
        ..burn_automata_kernels::AdaptivePerceptionConfig::growing_2d()
    };
    let options = burn_automata_kernels::AdaptiveNpaPerceptionOptions {
        eps0: model.config.eps0,
        scale_equivariance: model.config.scale_equivariant(),
        particle_density_equivariance: model.config.particle_density_equivariant(),
        log_norm_grad: model.config.log_norm_grad,
        log_norm_density_grad: model.config.log_norm_density_grad,
        position_features: model.config.position_features,
    };
    let run = || {
        rollout_adaptive_oracle_model_batch_chunk(
            &params,
            None,
            std::slice::from_ref(&target),
            &[0],
            x.clone(),
            s.clone(),
            material.clone(),
            bandwidth.clone(),
            Tensor::<BurnBackend, 2>::zeros([1, particle_count], &device),
            Tensor::<BurnBackend, 2>::zeros([1, particle_count], &device),
            perception,
            options,
            burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible,
            None,
            false,
            false,
            config,
            particle_count,
            &mut [canonical_eval_mask_rng(seed)],
            true,
            None,
            None,
            0,
            steps,
            Tensor::<BurnBackend, 1>::zeros([1], &device),
        )
    };
    let first = run();
    let second = run();
    assert_eq!(
        tensor3_vec(first.0.inner()).unwrap(),
        tensor3_vec(second.0.inner()).unwrap()
    );
    assert_eq!(
        tensor3_vec(first.1.inner()).unwrap(),
        tensor3_vec(second.1.inner()).unwrap()
    );
}

#[test]
fn adaptive_eval_mask_stream_uses_persistent_material_keys_and_weights() {
    let device = BurnDevice::default();
    let target = test_burn_target(&device, 0.5, 0.1);
    let seed = 47_u64;
    let mut first_keys = [0; 6];
    let mut first_weights = [0.0; 6];
    first_keys[0] = 1_234;
    first_weights[0] = 1.0;
    let mut second_keys = [0; 6];
    let mut second_weights = [0.0; 6];
    second_keys[..2].copy_from_slice(&[42, 43]);
    second_weights[..2].copy_from_slice(&[0.25, 0.75]);
    let mut expected_keys = [0; 6];
    let mut expected_weights = [0.0; 6];
    expected_keys[0] = 99;
    expected_weights[0] = 1.0;
    let masks = [
        crate::adaptive::AdaptiveTarget2dUpdateMask {
            keys: first_keys,
            weights: first_weights,
            expected: false,
        },
        crate::adaptive::AdaptiveTarget2dUpdateMask {
            keys: second_keys,
            weights: second_weights,
            expected: false,
        },
        crate::adaptive::AdaptiveTarget2dUpdateMask {
            keys: expected_keys,
            weights: expected_weights,
            expected: true,
        },
    ];
    let steps = 4;
    let actual = tensor3_vec(
        host_batch_wgpu_mask_stack(
            std::slice::from_ref(&target),
            &[0],
            3,
            steps,
            0,
            &[seed],
            Some(&masks),
        )
        .squeeze_dim::<3>(3)
        .inner(),
    )
    .unwrap();
    let random_seed = wgpu_random_seed(seed);
    let mut expected = Vec::with_capacity(steps * 3);
    for step in 0..steps {
        expected.push(f32::from(
            wgpu_random01(1_234, step as u32, random_seed) < 0.5,
        ));
        expected.push(
            0.25 * f32::from(wgpu_random01(42, step as u32, random_seed) < 0.5)
                + 0.75 * f32::from(wgpu_random01(43, step as u32, random_seed) < 0.5),
        );
        expected.push(0.5);
    }
    assert_eq!(actual, expected);
}

#[cfg(feature = "gpu_wgpu")]
#[test]
fn adaptive_burn_rollout_matches_wgpu_with_material_identity_masks() {
    let device = BurnDevice::default();
    let (rule_config, grid) =
        crate::NpaConfig::for_preset(crate::AutomataPreset::Growing2d);
    let rule = NpaModel::seeded(rule_config, 131);
    let mut adaptive_config = crate::AdaptiveNpaConfig::growing_2d();
    let reference_count = 16;
    let particle_count = 13;
    let fine_measure = std::f32::consts::PI * 0.2_f32.powi(2) / reference_count as f32;
    let fine_footprint = crate::material_footprint_radius(fine_measure, 2);
    adaptive_config.reference_footprint = fine_footprint;
    adaptive_config.base_rule_footprint = fine_footprint;
    adaptive_config.min_footprint = 0.5 * fine_footprint;
    adaptive_config.max_footprint = 4.0 * fine_footprint;
    adaptive_config.initial_leaves = particle_count;
    adaptive_config.min_leaves = particle_count;
    adaptive_config.target_leaves = particle_count;
    adaptive_config.max_leaves = reference_count;
    adaptive_config.bootstrap_fine_leaves = reference_count;
    adaptive_config.bootstrap_target_leaves = reference_count;
    adaptive_config.hierarchical_bootstrap_seed = true;
    adaptive_config.retain_bootstrap_templates = false;
    adaptive_config.rule_graph_policy =
        burn_automata_kernels::AdaptiveGraphPolicy::RawSupport;
    adaptive_config.perception.graph_policy =
        burn_automata_kernels::AdaptiveGraphPolicy::RawSupport;
    adaptive_config.perception.reference_measure = fine_measure;
    adaptive_config.perception.min_bandwidth = grid.eps;
    adaptive_config.perception.max_bandwidth = 2.0 * grid.eps;
    adaptive_config.perception.support_bin_ratio = 2.0;
    adaptive_config.proxy.enabled = false;
    adaptive_config.expected_coarse_update_mask = true;
    let adaptive_model =
        crate::AdaptiveNpaModel::seeded(rule, adaptive_config, 137).unwrap();
    let seed = 149_u64;
    let particles = crate::seed_adaptive_particles_scaled(
        &adaptive_model,
        particle_count,
        seed,
        crate::ParticleSeed::UniformCircle,
        0.2,
        fine_measure * reference_count as f32,
        grid.eps,
    )
    .unwrap();
    assert!(particles.bootstrap_templates.is_empty());
    assert!(
        particles
            .represented_measure
            .iter()
            .any(|measure| *measure > 1.5 * fine_measure)
    );
    let positions = particles
        .positions
        .iter()
        .flat_map(|position| [position[0], position[1]])
        .collect::<Vec<_>>();
    let represented_measure = particles.represented_measure.clone();
    let bandwidth = particles.bandwidth.clone();
    let states = particles.states.clone();
    let masks = particles
        .particle_id
        .iter()
        .copied()
        .enumerate()
        .map(|(row, particle_id)| {
            let mut keys = [0; 6];
            let mut weights = [0.0; 6];
            keys[0] = (particle_id as u32) ^ ((particle_id >> 32) as u32);
            weights[0] = 1.0;
            crate::adaptive::AdaptiveTarget2dUpdateMask {
                keys,
                weights,
                expected: represented_measure[row] > 1.5 * fine_measure,
            }
        })
        .collect::<Vec<_>>();

    let params =
        BurnBaseBatch::from_models(std::slice::from_ref(&adaptive_model.rule), &device)
            .unwrap()
            .detached();
    let mut config = test_direct_config(particle_count);
    config.update_prob = 0.5;
    config.grid_eps = grid.eps;
    config.motion_scale = adaptive_model.rule.config.alpha
        * adaptive_model.rule.config.motion_eps(grid.eps);
    let target = test_burn_target(&device, config.update_prob, 0.2);
    let steps = 3;
    let mut rngs = [canonical_eval_mask_rng(seed)];
    let (burn_positions, burn_states, _, _) =
        rollout_adaptive_oracle_model_batch_chunk(
            &params,
            None,
            std::slice::from_ref(&target),
            &[0],
            tensor3(positions, [1, particle_count, 2], &device),
            tensor3(
                states,
                [1, particle_count, adaptive_model.rule.config.state_dims],
                &device,
            ),
            tensor(
                represented_measure,
                [1, particle_count],
                &device,
            ),
            tensor(bandwidth, [1, particle_count], &device),
            Tensor::<BurnBackend, 2>::zeros([1, particle_count], &device),
            Tensor::<BurnBackend, 2>::zeros([1, particle_count], &device),
            adaptive_model.config.perception,
            burn_automata_kernels::AdaptiveNpaPerceptionOptions {
                eps0: adaptive_model.rule.config.eps0,
                scale_equivariance: adaptive_model.rule.config.scale_equivariant(),
                particle_density_equivariance: adaptive_model
                    .rule
                    .config
                    .particle_density_equivariant(),
                log_norm_grad: adaptive_model.rule.config.log_norm_grad,
                log_norm_density_grad: adaptive_model.rule.config.log_norm_density_grad,
                position_features: adaptive_model.rule.config.position_features,
            },
            burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible,
            None,
            false,
            false,
            config,
            particle_count,
            &mut rngs,
            true,
            Some(&[seed]),
            Some(&masks),
            0,
            steps,
            Tensor::<BurnBackend, 1>::zeros([1], &device),
        );

    let executor = crate::gpu::WgpuAutomataExecutor::new_blocking().unwrap();
    let mut wgpu_state = executor
        .create_adaptive_state(
            &adaptive_model,
            particles,
            &grid,
            1.0,
            crate::gpu::WgpuNeighborMode::CooperativeSortedCells,
            config.update_prob,
            seed,
        )
        .unwrap();
    executor.set_adaptive_stable_sorted_cells_enabled(&mut wgpu_state, true);
    executor
        .step_adaptive_state_many(&mut wgpu_state, steps, false)
        .unwrap();
    executor
        .synchronize_adaptive_particles(&mut wgpu_state)
        .unwrap();
    let expected_positions = wgpu_state
        .particles
        .positions
        .iter()
        .flat_map(|position| [position[0], position[1]])
        .collect::<Vec<_>>();
    let position_difference = max_abs_difference(
        &tensor3_vec(burn_positions.inner()).unwrap(),
        &expected_positions,
    );
    let state_difference = max_abs_difference(
        &tensor3_vec(burn_states.inner()).unwrap(),
        &wgpu_state.particles.states,
    );
    eprintln!(
        "adaptive Burn/WGPU material-mask parity: position={position_difference:e} state={state_difference:e}"
    );
    assert!(
        position_difference < 2.0e-4 && state_difference < 2.0e-4,
        "adaptive Burn/WGPU rollout mismatch with material identities: position={position_difference:e} state={state_difference:e}"
    );
}

#[test]
fn zero_normalized_adaptive_residual_preserves_the_compatible_base_rollout() {
    let device = BurnDevice::default();
    let base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 79);
    let mut residual_config = base.config.clone();
    residual_config.auxiliary_input_dims = 2;
    let residual = NpaModel {
        config: residual_config.clone(),
        weights: NpaWeights::zero_output_seeded(&residual_config, 83),
    };
    let base_params =
        BurnBaseBatch::from_models(std::slice::from_ref(&base), &device).unwrap().detached();
    let residual_params =
        BurnBaseBatch::from_models(std::slice::from_ref(&residual), &device).unwrap();
    let particle_count = 8;
    let steps = 4;
    let mut config = test_direct_config(particle_count);
    config.update_prob = 1.0;
    config.rollout_steps = steps;
    config.tbptt_chunk_steps = steps;
    let target = test_burn_target(&device, 1.0, 0.1);
    let (x, s) = seed_tensors(particle_count, config, 0.1, 89, &device);
    let x = x.reshape([1, particle_count, 2]);
    let s = s.reshape([1, particle_count, base.config.state_dims]);
    let represented_measure = tensor(
        vec![4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1, particle_count],
        &device,
    );
    let bandwidth = tensor(
        vec![0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [1, particle_count],
        &device,
    );
    let residual_gate = tensor(
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1, particle_count],
        &device,
    );
    let material_scale_feature = tensor(
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1, particle_count],
        &device,
    );
    let perception = burn_automata_kernels::AdaptivePerceptionConfig {
        graph_policy: burn_automata_kernels::AdaptiveGraphPolicy::RawSupport,
        max_neighbors: particle_count,
        min_bandwidth: 0.1,
        max_bandwidth: 0.2,
        ..burn_automata_kernels::AdaptivePerceptionConfig::growing_2d()
    };
    let options = burn_automata_kernels::AdaptiveNpaPerceptionOptions {
        eps0: base.config.eps0,
        scale_equivariance: base.config.scale_equivariant(),
        particle_density_equivariance: base.config.particle_density_equivariant(),
        log_norm_grad: base.config.log_norm_grad,
        log_norm_density_grad: base.config.log_norm_density_grad,
        position_features: base.config.position_features,
    };
    let mut baseline_rngs = [StdRng::seed_from_u64(97)];
    let baseline = rollout_adaptive_oracle_model_batch_chunk(
        &base_params,
        None,
        std::slice::from_ref(&target),
        &[0],
        x.clone(),
        s.clone(),
        represented_measure.clone(),
        bandwidth.clone(),
        material_scale_feature.clone(),
        residual_gate.clone(),
        perception,
        options,
        burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible,
        None,
        false,
        false,
        config,
        particle_count,
        &mut baseline_rngs,
        true,
        None,
        None,
        0,
        steps,
        Tensor::<BurnBackend, 1>::zeros([1], &device),
    );
    let mut residual_rngs = [StdRng::seed_from_u64(97)];
    let with_zero_residual = rollout_adaptive_oracle_model_batch_chunk(
        &residual_params,
        Some(&base_params),
        std::slice::from_ref(&target),
        &[0],
        x,
        s,
        represented_measure,
        bandwidth,
        material_scale_feature,
        residual_gate,
        perception,
        options,
        burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible,
        Some(burn_automata_kernels::AdaptivePerceptionSemantics::NormalizedAdaptive),
        false,
        true,
        config,
        particle_count,
        &mut residual_rngs,
        true,
        None,
        None,
        0,
        steps,
        Tensor::<BurnBackend, 1>::zeros([1], &device),
    );
    assert_eq!(
        tensor3_vec(baseline.0.inner()).unwrap(),
        tensor3_vec(with_zero_residual.0.inner()).unwrap()
    );
    assert_eq!(
        tensor3_vec(baseline.1.inner()).unwrap(),
        tensor3_vec(with_zero_residual.1.inner()).unwrap()
    );
}

#[test]
fn compatible_residual_receives_gradient_at_fixed_support() {
    let device = BurnDevice::default();
    let base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 101);
    let mut residual_config = base.config.clone();
    residual_config.auxiliary_input_dims = 2;
    let residual = NpaModel {
        config: residual_config.clone(),
        weights: NpaWeights::zero_output_seeded(&residual_config, 103),
    };
    let base_params =
        BurnBaseBatch::from_models(std::slice::from_ref(&base), &device).unwrap().detached();
    let residual_params =
        BurnBaseBatch::from_models(std::slice::from_ref(&residual), &device).unwrap();
    let particle_count = 8;
    let mut config = test_direct_config(particle_count);
    config.motion_scale = 0.0;
    config.update_prob = 1.0;
    config.rollout_steps = 1;
    config.tbptt_chunk_steps = 1;
    let target = test_burn_target(&device, 1.0, 0.1);
    let (x, s) = seed_tensors(particle_count, config, 0.1, 107, &device);
    let perception = burn_automata_kernels::AdaptivePerceptionConfig {
        graph_policy: burn_automata_kernels::AdaptiveGraphPolicy::RawSupport,
        max_neighbors: particle_count,
        reference_measure: 1.0,
        min_bandwidth: 0.1,
        max_bandwidth: 0.1,
        ..burn_automata_kernels::AdaptivePerceptionConfig::growing_2d()
    };
    let options = burn_automata_kernels::AdaptiveNpaPerceptionOptions {
        eps0: base.config.eps0,
        scale_equivariance: base.config.scale_equivariant(),
        particle_density_equivariance: base.config.particle_density_equivariant(),
        log_norm_grad: base.config.log_norm_grad,
        log_norm_density_grad: base.config.log_norm_density_grad,
        position_features: base.config.position_features,
    };
    let (_, states, _, _) = rollout_adaptive_oracle_model_batch_chunk(
        &residual_params,
        Some(&base_params),
        std::slice::from_ref(&target),
        &[0],
        x.reshape([1, particle_count, 2]),
        s.reshape([1, particle_count, base.config.state_dims]),
        tensor(
            vec![4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1, particle_count],
            &device,
        ),
        tensor(vec![0.1; particle_count], [1, particle_count], &device),
        tensor(
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1, particle_count],
            &device,
        ),
        tensor(
            vec![1.0; particle_count],
            [1, particle_count],
            &device,
        ),
        perception,
        options,
        burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible,
        Some(burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible),
        false,
        true,
        config,
        particle_count,
        &mut [StdRng::seed_from_u64(109)],
        true,
        None,
        None,
        0,
        1,
        Tensor::<BurnBackend, 1>::zeros([1], &device),
    );
    let mut gradients = states.sum().backward();
    let w2_gradient = test_tensor_values(
        residual_params
            .w2
            .grad_remove(&mut gradients)
            .expect("fixed-support coarse residual must receive a w2 gradient"),
    );
    let gradient_norm = w2_gradient
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!(
        gradient_norm > 1.0e-8,
        "fixed-support represented-measure residual gradient was zero"
    );
}

#[test]
fn adaptive_shared_rule_receives_the_per_row_material_scale_feature() {
    let device = BurnDevice::default();
    let particle_count = 2;
    let mut npa = NpaConfig::growing_2d();
    npa.auxiliary_input_dims = 1;
    let mut weights = NpaWeights::zeros(&npa);
    let material_scale_input = npa.perception_dims() - 1;
    weights.w1[material_scale_input] = 1.0;
    let first_state_output = npa.spatial_dims;
    weights.w2[first_state_output * npa.hidden_dims] = 1.0;
    let model = NpaModel {
        config: npa.clone(),
        weights,
    };
    let params = BurnBaseBatch::from_models(std::slice::from_ref(&model), &device).unwrap();
    let mut config = test_direct_config(particle_count);
    config.motion_scale = 0.0;
    config.update_prob = 1.0;
    let target = test_burn_target(&device, 1.0, 0.1);
    let perception = burn_automata_kernels::AdaptivePerceptionConfig {
        graph_policy: burn_automata_kernels::AdaptiveGraphPolicy::RawSupport,
        max_neighbors: particle_count,
        min_bandwidth: 0.1,
        max_bandwidth: 0.2,
        ..burn_automata_kernels::AdaptivePerceptionConfig::growing_2d()
    };
    let options = burn_automata_kernels::AdaptiveNpaPerceptionOptions {
        eps0: 0.1,
        scale_equivariance: npa.scale_equivariant(),
        particle_density_equivariance: npa.particle_density_equivariant(),
        log_norm_grad: npa.log_norm_grad,
        log_norm_density_grad: npa.log_norm_density_grad,
        position_features: npa.position_features,
    };
    let (_, states, _, _) = rollout_adaptive_oracle_model_batch_chunk(
        &params,
        None,
        std::slice::from_ref(&target),
        &[0],
        Tensor::<BurnBackend, 3>::zeros([1, particle_count, 2], &device),
        Tensor::<BurnBackend, 3>::zeros([1, particle_count, npa.state_dims], &device),
        tensor(vec![1.0, 4.0], [1, particle_count], &device),
        tensor(vec![0.1, 0.1], [1, particle_count], &device),
        tensor(vec![0.0, 1.0], [1, particle_count], &device),
        Tensor::<BurnBackend, 2>::zeros([1, particle_count], &device),
        perception,
        options,
        burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible,
        None,
        true,
        false,
        config,
        particle_count,
        &mut [StdRng::seed_from_u64(73)],
        true,
        None,
        None,
        0,
        1,
        Tensor::<BurnBackend, 1>::zeros([1], &device),
    );
    let states = tensor3_vec(states.inner()).unwrap();
    assert!(states[0].abs() <= 1.0e-6);
    assert!((states[npa.state_dims] - 1.0).abs() <= 1.0e-6);
}

#[test]
fn adaptive_motion_uses_each_rows_scale_equivariant_bandwidth() {
    let device = BurnDevice::default();
    let raw = tensor3(vec![3.0, 4.0, 0.0, 2.0], [1, 2, 2], &device);
    let bandwidth = tensor(vec![0.1, 0.2], [1, 2], &device);
    let (motion, squared, denominator, row_scale) = normalized_adaptive_motion_batch(
        raw,
        0.02,
        bandwidth,
        0.1,
        true,
        1,
        2,
    );
    let motion = tensor3_vec(motion.inner()).unwrap();
    let expected = [0.01, 0.013_333_334, 0.0, 0.026_666_667];
    assert!(
        max_abs_difference(&motion, &expected) <= 1.0e-6,
        "adaptive per-row motion scale mismatch: {motion:?}"
    );
    let displacement =
        adaptive_displacement_magnitude_batch(squared, denominator, row_scale);
    let displacement = tensor3_vec(displacement.inner()).unwrap();
    assert!((displacement[0] - (0.01_f32.powi(2) + 0.013_333_334_f32.powi(2)).sqrt()).abs() <= 1.0e-6);
    assert!((displacement[1] - 0.026_666_667).abs() <= 1.0e-6);
}

#[test]
fn adaptive_checkpoint_selection_prioritizes_render_psnr() {
    assert!(adaptive_training::adaptive_checkpoint_is_better(18.0, None));
    assert!(adaptive_training::adaptive_checkpoint_is_better(
        18.0,
        Some(17.9)
    ));
    assert!(!adaptive_training::adaptive_checkpoint_is_better(
        17.8,
        Some(17.9)
    ));
    assert!(!adaptive_training::adaptive_checkpoint_is_better(
        f32::NAN,
        Some(17.9)
    ));
}

#[test]
fn adaptive_target2d_primal_parity_is_rowwise_and_fail_closed() {
    adaptive_training::validate_adaptive_target2d_primal_parity(
        &[0.0041, 0.0021, 0.0019],
        &[0.0040, 0.0020, 0.0020],
        512,
    )
    .unwrap();

    let tail = adaptive_training::validate_adaptive_target2d_primal_parity(
        &[0.0041, 0.0021, 10.0],
        &[0.0040, 0.0020, 0.0020],
        512,
    )
    .unwrap_err();
    assert!(tail.to_string().contains("row 2"));

    let non_finite = adaptive_training::validate_adaptive_target2d_primal_parity(
        &[0.0041, f32::NAN],
        &[0.0040, 0.0020],
        96,
    )
    .unwrap_err();
    assert!(non_finite.to_string().contains("non-finite"));
}

#[test]
fn functional_teacher_probe_loss_distinguishes_adapter_behavior() {
    let device = BurnDevice::default();
    let config = NpaConfig::growing_2d();
    let base = NpaModel::upstream_seeded(config.clone(), 7);
    let params = BurnBaseParams::from_model(&base, &device).unwrap();
    let rank = 4;
    let mut adapter = NpaLowRankAdapter::seeded(&config, rank, rank as f32, 19);
    adapter.b2_delta.fill(0.25);
    let adapter_vector = adapter.to_parameter_vector();
    let adapter_batch = BurnAdapterBatch::from_parameter_vector(
        tensor(adapter_vector, [1, adapter.parameter_count()], &device),
        &config,
        rank,
        rank as f32,
    );
    let zero_batch = BurnAdapterBatch::from_parameter_vector(
        Tensor::<BurnBackend, 2>::zeros([1, adapter.parameter_count()], &device),
        &config,
        rank,
        rank as f32,
    );
    let feature_dims = config.perception_dims();
    let features = tensor3(
        (0..3 * feature_dims)
            .map(|idx| (idx as f32 * 0.071).sin())
            .collect(),
        [1, 3, feature_dims],
        &device,
    );
    let teacher = params.forward_adapter_batch(features.clone(), &adapter_batch);
    let matching = params.forward_adapter_batch(features.clone(), &adapter_batch);
    let mismatched = params.forward_adapter_batch(features, &zero_batch);
    let matching_delta = matching - teacher.clone();
    let mismatched_delta = mismatched - teacher;
    let matching_mse = matching_delta
        .clone()
        .mul(matching_delta)
        .mean()
        .inner()
        .into_scalar();
    let mismatched_mse = mismatched_delta
        .clone()
        .mul(mismatched_delta)
        .mean()
        .inner()
        .into_scalar();

    assert!(matching_mse <= 1.0e-12);
    assert!(mismatched_mse > 1.0e-8);
}

#[test]
fn functional_amortization_distillation_only_gradients_generated_adapter() {
    let device = BurnDevice::default();
    let config = NpaConfig::growing_2d();
    let base = NpaModel::upstream_seeded(config.clone(), 23);
    let params = BurnBaseParams::from_model(&base, &device).unwrap();
    let rank = 4;
    let mut generated = NpaLowRankAdapter::seeded(&config, rank, rank as f32, 29);
    generated.b1_delta.fill(0.1);
    let generated_values = generated.to_parameter_vector();
    let parameter_count = generated_values.len();
    let generated_vector = tensor(generated_values, [1, parameter_count], &device).require_grad();
    let generated_probe = generated_vector.clone();
    let endpoint_vector =
        Tensor::<BurnBackend, 2>::zeros([1, parameter_count], &device).require_grad();
    let endpoint_probe = endpoint_vector.clone();
    let generated_adapter =
        BurnAdapterBatch::from_parameter_vector(generated_vector, &config, rank, rank as f32);
    let endpoint_adapter =
        BurnAdapterBatch::from_parameter_vector(endpoint_vector, &config, rank, rank as f32);
    let feature_dims = config.perception_dims();
    let probes = tensor3(
        (0..4 * feature_dims)
            .map(|index| (index as f32 * 0.037).sin())
            .collect(),
        [1, 4, feature_dims],
        &device,
    );
    let loss = functional_adapter_distillation_loss(
        &params,
        probes,
        &generated_adapter,
        &endpoint_adapter,
    );
    assert!(loss.clone().inner().into_scalar() > 1.0e-8);
    let mut grads = loss.backward();
    let generated_gradient = tensor_vec(
        generated_probe
            .grad_remove(&mut grads)
            .expect("generated adapter receives functional gradient"),
    )
    .unwrap();
    assert!(generated_gradient.iter().any(|value| value.abs() > 1.0e-9));
    assert!(endpoint_probe.grad_remove(&mut grads).is_none());
}

#[test]
fn spatial_condition_state_projection_is_zero_warm_start_compatible() {
    let device = BurnDevice::default();
    let x = tensor3(vec![0.0, 0.0], [1, 1, 2], &device);
    let state = tensor3(vec![1.0, 0.0], [1, 1, 2], &device);
    let base = BurnE2eConditionControlBatch {
        patch_hidden: tensor3(vec![1.0, 0.0], [1, 1, 2], &device),
        update_w: tensor(vec![1.0, 0.0, 0.0, 1.0], [2, 2], &device),
        update_b: tensor(vec![0.0, 0.0], [1, 2], &device),
        state_w: None,
        grid_width: 1,
        grid_height: 1,
        sigma: 0.25,
        scale: 1.0,
    };
    let baseline = tensor3_vec(base.update_for_particles(&x, &state).inner()).unwrap();
    let zero_state = BurnE2eConditionControlBatch {
        state_w: Some(tensor(vec![0.0; 4], [2, 2], &device)),
        ..base.clone()
    };
    let zero_state_output =
        tensor3_vec(zero_state.update_for_particles(&x, &state).inner()).unwrap();
    assert_eq!(baseline, zero_state_output);

    let state_aware = BurnE2eConditionControlBatch {
        state_w: Some(tensor(vec![1.0, 0.0, 0.0, 1.0], [2, 2], &device)),
        ..base
    };
    let state_aware_output =
        tensor3_vec(state_aware.update_for_particles(&x, &state).inner()).unwrap();
    assert!(state_aware_output[0] > baseline[0]);
}

#[test]
fn perception_auto_uses_fused_path_for_training_scale_particles() {
    let mut small = test_direct_config(127);
    small.perception_backend = PerceptionRolloutBackend::Auto;
    assert_eq!(
        perception_backend_effective(small),
        PerceptionRolloutBackend::Dense
    );

    let mut training_scale = test_direct_config(128);
    training_scale.perception_backend = PerceptionRolloutBackend::Auto;
    let expected = if PERCEPTION_CUBE_ENABLED {
        PerceptionRolloutBackend::TiledAdjoint
    } else {
        PerceptionRolloutBackend::Dense
    };
    assert_eq!(perception_backend_effective(training_scale), expected);
}

#[test]
fn target2d_auto_uses_device_adjoint_for_training_scale_particles() {
    let mut small = test_direct_config(127);
    small.target2d_loss_backend = Target2dLossBackend::Auto;
    assert_eq!(
        target2d_loss_backend_effective(small),
        Target2dLossBackend::Dense
    );

    let mut training_scale = test_direct_config(128);
    training_scale.target2d_loss_backend = Target2dLossBackend::Auto;
    let expected = if PERCEPTION_CUBE_ENABLED {
        Target2dLossBackend::TiledAdjoint
    } else {
        Target2dLossBackend::Dense
    };
    assert_eq!(target2d_loss_backend_effective(training_scale), expected);
}

#[test]
fn target2d_splat_denominator_is_translation_invariant_before_image_clipping() {
    let device = BurnDevice::default();
    let particle_count = 2;
    let mut config = test_direct_config(particle_count);
    config.loss_config.image_size = 32;
    let particle_pixels = tensor3(
        vec![15.25, 10.75, 37.25, 10.75],
        [1, particle_count, 2],
        &device,
    );
    let sigma = tensor3(vec![1.6, 1.6], [1, 1, particle_count], &device);
    let denominator =
        splat_particle_denominator_batch(&particle_pixels, particle_count, sigma, config);
    let values = tensor3_vec(denominator.inner()).unwrap();

    assert!(
        (values[0] - values[1]).abs() <= 1.0e-5,
        "full-patch Gaussian normalization changed after an integer translation outside the image: {values:?}"
    );
}

#[test]
fn adaptive_backward_scaling_bounds_large_linear_adjoint_surrogates() {
    assert_eq!(adaptive_backward_scale(1.0, true), 1.0);
    assert_eq!(adaptive_backward_scale(1.0e-4, true), 1.0e-4);
    assert_eq!(adaptive_backward_scale(1.0e-4, false), 1.0);
}

#[test]
fn stochastic_rollout_step_sampler_matches_upstream_exclusive_maximum() {
    let mut config = test_direct_config(4);
    config.rollout_step_min = 2;
    config.rollout_steps = 4;
    let samples = (0..512)
        .map(|seed| sampled_training_rollout_steps(config, seed))
        .collect::<Vec<_>>();
    assert!(
        samples.iter().all(|steps| (2..4).contains(steps)),
        "sampled rollout steps escaped configured upstream range: {samples:?}"
    );
    assert!(
        samples.contains(&2) && samples.contains(&3) && !samples.contains(&4),
        "sampled rollout steps did not match the upstream-exclusive maximum: {samples:?}"
    );
}

#[test]
fn production_rollout_step_sampler_covers_32_through_95() {
    let mut config = test_direct_config(4);
    config.rollout_step_min = 32;
    config.rollout_steps = 96;
    let samples = (0..8192)
        .map(|seed| sampled_training_rollout_steps(config, seed))
        .collect::<Vec<_>>();
    assert!(samples.iter().all(|steps| (32..96).contains(steps)));
    assert_eq!(samples.iter().copied().min(), Some(32));
    assert_eq!(samples.iter().copied().max(), Some(95));
}

#[test]
fn pre_rollout_sampler_covers_early_and_late_burn_in_states() {
    let samples = (0..8192)
        .map(|seed| sampled_pre_rollout_steps(0, 448, seed))
        .collect::<Vec<_>>();
    assert!(samples.iter().all(|steps| *steps < 448));
    assert_eq!(samples.iter().copied().min(), Some(0));
    assert_eq!(samples.iter().copied().max(), Some(447));
    assert_eq!(sampled_pre_rollout_steps(160, 160, 7), 160);
}

#[test]
fn rollout_quality_diagnostics_measure_occupancy_and_overflow() {
    let device = BurnDevice::default();
    let npa_config = NpaConfig::growing_2d();
    let rank = 2;
    let adapter = NpaLowRankAdapter::seeded_zero_delta(&npa_config, rank, rank as f32, 17);
    let adapter_batch = BurnAdapterBatch::from_parameter_vector(
        tensor(
            adapter.to_parameter_vector(),
            [1, adapter.parameter_count()],
            &device,
        ),
        &npa_config,
        rank,
        rank as f32,
    );
    let target = test_burn_target(&device, 1.0, 0.1);
    let mut config = test_direct_config(4);
    config.loss_config.image_size = 1;
    config.loss_config.center = false;
    let x = tensor3(
        vec![1.1, 0.0, -1.2, 1.3, 0.0, 0.0, 0.5, -0.5],
        [1, 4, 2],
        &device,
    );
    let mut state_values = vec![0.0; 4 * npa_config.state_dims];
    state_values[0] = 1.1;
    state_values[17] = -2.0;
    let s = tensor3(state_values, [1, 4, npa_config.state_dims], &device);
    let quality = target_splat_quality_batch_vector(
        &x,
        &s,
        &[target],
        &[0],
        config,
        &adapter_batch,
        Tensor::<BurnBackend, 1>::zeros([1], &device),
    );
    let occupancy = tensor1_vec(quality.render_occupancy.inner()).unwrap()[0];
    let position_overflow = tensor1_vec(quality.position_overflow_fraction.inner()).unwrap()[0];
    let state_overflow = tensor1_vec(quality.state_overflow_fraction.inner()).unwrap()[0];
    assert!((0.0..=1.0).contains(&occupancy));
    assert!((position_overflow - 0.5).abs() < 1.0e-6);
    assert!((state_overflow - 2.0 / 64.0).abs() < 1.0e-6);
}

#[test]
fn brush_centers_are_gathered_from_live_particles_per_batch_row() {
    let device = BurnDevice::default();
    let positions = tensor3(
        vec![
            -0.9, -0.8, -0.4, -0.3, 0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 1.0,
        ],
        [2, 3, 2],
        &device,
    );
    let centers = gather_live_particle_centers(positions, &[2, 0], &device);
    assert_eq!(tensor3_vec(centers.inner()).unwrap(), [0.1, 0.2, 0.3, 0.4]);
}

#[test]
fn e2e_generator_zero_delta_lora_init_keeps_adapter_trainable() {
    let config = NpaConfig::growing_2d();
    let rank = 4;
    let output_scale = 1.0;
    let bias = seeded_zero_delta_output_bias(&config, rank, rank as f32, 123, output_scale);
    let vector = bias
        .iter()
        .map(|value| value.tanh() * output_scale)
        .collect::<Vec<_>>();
    let adapter =
        NpaLowRankAdapter::from_parameter_vector(&config, rank, rank as f32, vector).unwrap();
    assert!(
        adapter
            .w1_down
            .iter()
            .chain(adapter.w2_down.iter())
            .any(|value| value.abs() > 1.0e-6),
        "zero-delta LoRA init must seed one side of each low-rank product"
    );
    assert!(
        adapter
            .w1_up
            .iter()
            .chain(adapter.w2_up.iter())
            .chain(adapter.b1_delta.iter())
            .chain(adapter.b2_delta.iter())
            .all(|value| value.abs() <= 1.0e-6),
        "zero-delta LoRA init must not perturb the base model initially"
    );
}

#[test]
fn canonical_lora_batched_device_transform_matches_shared_cpu_layout() {
    let device = BurnDevice::default();
    let config = NpaConfig::growing_2d();
    let rank = config.perception_dims().max(config.update_dims());
    let canonical =
        crate::hyper::adapter_layout::CanonicalFullRankLora2d::new(&config, rank, rank as f32)
            .unwrap();
    let output_dims = canonical.constants.len();
    let first = (0..output_dims)
        .map(|index| (index as f32 * 0.013).sin() * 0.01)
        .collect::<Vec<_>>();
    let second = (0..output_dims)
        .map(|index| (index as f32 * 0.017).cos() * 0.02)
        .collect::<Vec<_>>();
    let expected = [
        canonical.apply(&first).unwrap(),
        canonical.apply(&second).unwrap(),
    ]
    .concat();
    let generated = tensor([first, second].concat(), [2, output_dims], &device);
    let mask = tensor(canonical.trainable_mask.clone(), [1, output_dims], &device)
        .expand([2, output_dims]);
    let constants = tensor(canonical.constants, [1, output_dims], &device).expand([2, output_dims]);
    let actual = tensor_vec((generated.mul(mask) + constants).inner()).unwrap();

    assert!(max_abs_difference(&actual, &expected) < 1.0e-7);
    assert_ne!(&actual[..output_dims], &actual[output_dims..]);
}

#[test]
fn hyper_e2e_batch_dimension_is_sample_parallel() {
    let device = BurnDevice::default();
    let targets = vec![
        test_burn_target(&device, 1.0, 0.1),
        test_burn_target(&device, 0.0, 0.2),
    ];
    let indices = [0usize, 1usize];
    let particle_count = 4;
    let (x, s) = seed_batch_tensors(
        &targets,
        &indices,
        particle_count,
        test_direct_config(particle_count),
        77,
        &device,
    );
    assert_eq!(x.shape().dims::<3>(), [2, particle_count, 2]);
    assert_eq!(s.shape().dims::<3>(), [2, particle_count, 16]);
    let x_values = tensor3_vec(x.inner()).unwrap();
    assert_ne!(
        &x_values[0..particle_count * 2],
        &x_values[particle_count * 2..particle_count * 4],
        "seeded rollout batch collapsed two independent samples into one state"
    );

    let mask = host_batch_mask_seeded(&targets, &indices, particle_count, 123);
    assert_eq!(mask.shape().dims::<3>(), [2, particle_count, 1]);
    let mask_values = tensor3_vec(mask.inner()).unwrap();
    assert!(
        mask_values[0..particle_count]
            .iter()
            .all(|value| *value == 1.0)
    );
    assert!(
        mask_values[particle_count..particle_count * 2]
            .iter()
            .all(|value| *value == 0.0)
    );
    let device_mask = device_batch_mask_stack(&targets, &indices, particle_count, 2);
    assert_eq!(device_mask.shape().dims::<4>(), [2, 2, particle_count, 1]);
    let device_mask_values = tensor3_vec(
        device_mask
            .reshape([2 * indices.len(), particle_count, 1])
            .inner(),
    )
    .unwrap();
    assert!(
        device_mask_values[0..particle_count]
            .iter()
            .all(|value| *value == 1.0),
        "update_prob=1.0 should keep all device-mask entries active"
    );
    assert!(
        device_mask_values[particle_count..particle_count * 2]
            .iter()
            .all(|value| *value == 0.0),
        "update_prob=0.0 should keep all device-mask entries inactive"
    );

    let npa_config = NpaConfig::growing_2d();
    let rank = 2;
    let parameter_count = NpaLowRankAdapter::parameter_count_for_config(&npa_config, rank);
    let mut vector = Vec::with_capacity(parameter_count * 2);
    vector.extend(std::iter::repeat_n(0.0, parameter_count));
    vector.extend(std::iter::repeat_n(1.0, parameter_count));
    let adapter_batch = BurnAdapterBatch::from_parameter_vector(
        tensor(vector, [2, parameter_count], &device),
        &npa_config,
        rank,
        1.0,
    );
    assert_eq!(adapter_batch.w1_down.shape().dims::<3>()[0], 2);
    let w1_down = tensor3_vec(adapter_batch.w1_down.inner()).unwrap();
    let row_len = rank * npa_config.perception_dims();
    assert!(w1_down[0..row_len].iter().all(|value| *value == 0.0));
    assert!(
        w1_down[row_len..row_len * 2]
            .iter()
            .all(|value| *value == 1.0)
    );

    let model = NpaModel::seeded(npa_config.clone(), 91);
    let params = BurnBaseParams::from_model(&model, &device).unwrap();
    let host_adapters = [
        NpaLowRankAdapter::seeded(&npa_config, rank, 1.0, 101),
        NpaLowRankAdapter::seeded(&npa_config, rank, 1.0, 202),
    ];
    let burn_adapters = host_adapters
        .iter()
        .map(|adapter| BurnAdapterParams::from_adapter(adapter, &model, &device).unwrap())
        .collect::<Vec<_>>();
    let batch_adapter = BurnAdapterBatch::from_indices(&burn_adapters, &[0, 1]);
    let expanded_adapter = batch_adapter.clone().select_rows(&[0, 0, 1, 1]);
    let expanded_w1 = tensor3_vec(expanded_adapter.w1_down.inner()).unwrap();
    let adapter_row = rank * npa_config.perception_dims();
    assert_eq!(
        &expanded_w1[..adapter_row],
        &expanded_w1[adapter_row..2 * adapter_row]
    );
    assert_eq!(
        &expanded_w1[2 * adapter_row..3 * adapter_row],
        &expanded_w1[3 * adapter_row..4 * adapter_row]
    );
    assert_ne!(
        &expanded_w1[..adapter_row],
        &expanded_w1[2 * adapter_row..3 * adapter_row]
    );
    let rows = 3;
    let input_dims = npa_config.perception_dims();
    let feature_values = (0..2 * rows * input_dims)
        .map(|index| index as f32 * 0.001 - 0.25)
        .collect::<Vec<_>>();
    let batch_output = params.forward_adapter_batch(
        tensor3(feature_values.clone(), [2, rows, input_dims], &device),
        &batch_adapter,
    );
    let batch_values = tensor3_vec(batch_output.inner()).unwrap();
    let output_dims = npa_config.update_dims();
    for (batch, burn_adapter) in burn_adapters.iter().enumerate().take(2) {
        let feature_start = batch * rows * input_dims;
        let single_output = params.forward_adapter(
            tensor(
                feature_values[feature_start..feature_start + rows * input_dims].to_vec(),
                [rows, input_dims],
                &device,
            ),
            burn_adapter,
            test_direct_config(particle_count),
        );
        let single_values = tensor_vec(single_output.inner()).unwrap();
        let batch_start = batch * rows * output_dims;
        for (actual, expected) in batch_values[batch_start..batch_start + rows * output_dims]
            .iter()
            .zip(single_values)
        {
            assert!(
                (actual - expected).abs() < 1.0e-5,
                "batched per-sample LoRA output diverged from unbatched output: {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn reused_motion_norm_matches_direct_displacement_value_and_gradient() {
    let device = BurnDevice::default();
    let values = vec![0.2, -0.4, 0.7, 0.1, -0.3, -0.8, 0.6, 0.5];
    let scale = 0.125;
    let raw_reused = tensor3(values.clone(), [2, 2, 2], &device).require_grad();
    let raw_direct = tensor3(values, [2, 2, 2], &device).require_grad();
    let (dx_reused, squared, denominator) =
        normalized_motion_batch(raw_reused.clone(), scale, 2, 2);
    let reused = displacement_magnitude_batch(squared, denominator, scale);
    let direct = dx_reused
        .clone()
        .mul(dx_reused)
        .sum_dim(2)
        .add_scalar(EPSILON * EPSILON)
        .sqrt();
    let reused_values = tensor3_vec(reused.clone().inner()).unwrap();
    let direct_values = tensor3_vec(direct.clone().inner()).unwrap();
    assert!(max_abs_difference(&reused_values, &direct_values) < 1.0e-7);

    let mut reused_grads = reused.sum().backward();
    let reused_gradient = tensor3_vec(
        raw_reused
            .grad_remove(&mut reused_grads)
            .expect("reused displacement gradient"),
    )
    .unwrap();
    let (dx_direct, _, _) = normalized_motion_batch(raw_direct.clone(), scale, 2, 2);
    let direct_objective = dx_direct
        .clone()
        .mul(dx_direct)
        .sum_dim(2)
        .add_scalar(EPSILON * EPSILON)
        .sqrt()
        .sum();
    let mut direct_grads = direct_objective.backward();
    let direct_gradient = tensor3_vec(
        raw_direct
            .grad_remove(&mut direct_grads)
            .expect("direct displacement gradient"),
    )
    .unwrap();
    assert!(max_abs_difference(&reused_gradient, &direct_gradient) < 1.0e-6);
}

#[test]
fn canonical_dense_residual_matches_generic_lora_expansion() {
    let device = BurnDevice::default();
    let npa = NpaConfig::growing_2d();
    let mut model = NpaModel::upstream_seeded(npa.clone(), 0x91a7);
    model.weights.b1.fill(0.25);
    let conditions = 2;
    let replicas = 2;
    let particles = 5;
    let batches = conditions * replicas;
    let layout = NpaParameterRowLayout2d::new(&npa);
    let row_values = (0..conditions * layout.row_count() * layout.max_row_dims())
        .map(|index| (index as f32 * 0.013).sin() * 0.01)
        .collect::<Vec<_>>();
    let feature_values = (0..batches * particles * npa.perception_dims())
        .map(|index| (index as f32 * 0.019).cos() * 0.2)
        .collect::<Vec<_>>();
    let objective_weights = (0..batches * particles * npa.update_dims())
        .map(|index| (index as f32 * 0.029).sin())
        .collect::<Vec<_>>();
    let expansion = vec![0, 0, 1, 1];

    let grouped_rows = tensor3(
        row_values.clone(),
        [conditions, layout.row_count(), layout.max_row_dims()],
        &device,
    )
    .require_grad();
    let grouped_features = tensor3(
        feature_values.clone(),
        [batches, particles, npa.perception_dims()],
        &device,
    )
    .require_grad();
    let grouped_params = BurnBaseParams::from_model(&model, &device).unwrap();
    let grouped_adapter = BurnAdapterBatch::from_dense_residual_rows(grouped_rows.clone(), &npa)
        .select_rows(&expansion);
    assert_eq!(grouped_adapter.w1_up.shape().dims::<3>()[0], batches);
    let grouped_output =
        grouped_params.forward_adapter_batch(grouped_features.clone(), &grouped_adapter);
    let weights = tensor3(
        objective_weights.clone(),
        [batches, particles, npa.update_dims()],
        &device,
    );
    let mut grouped_grads = grouped_output.clone().mul(weights).sum().backward();

    let explicit_rows = tensor3(
        row_values,
        [conditions, layout.row_count(), layout.max_row_dims()],
        &device,
    )
    .require_grad();
    let explicit_features = tensor3(
        feature_values,
        [batches, particles, npa.perception_dims()],
        &device,
    )
    .require_grad();
    let explicit_params = BurnBaseParams::from_model(&model, &device).unwrap();
    let mut explicit_adapter =
        BurnAdapterBatch::from_dense_residual_rows(explicit_rows.clone(), &npa);
    explicit_adapter.canonical_dense_residual = false;
    let explicit_adapter = explicit_adapter.select_rows(&expansion);
    let weights = tensor3(
        objective_weights,
        [batches, particles, npa.update_dims()],
        &device,
    );
    let explicit_output =
        explicit_params.forward_adapter_batch(explicit_features.clone(), &explicit_adapter);
    let mut explicit_grads = explicit_output.clone().mul(weights).sum().backward();

    assert!(
        max_abs_difference(
            &tensor3_vec(grouped_output.inner()).unwrap(),
            &tensor3_vec(explicit_output.inner()).unwrap(),
        ) < 2.0e-4
    );
    macro_rules! compare_grouped_grad {
        ($grouped:expr, $explicit:expr, $tolerance:expr) => {{
            let grouped = test_tensor_values(
                $grouped
                    .grad_remove(&mut grouped_grads)
                    .expect("grouped gradient"),
            );
            let explicit = test_tensor_values(
                $explicit
                    .grad_remove(&mut explicit_grads)
                    .expect("explicit gradient"),
            );
            assert!(max_abs_difference(&grouped, &explicit) < $tolerance);
        }};
    }
    compare_grouped_grad!(grouped_features, explicit_features, 3.0e-4);
    compare_grouped_grad!(grouped_rows, explicit_rows, 3.0e-3);
    compare_grouped_grad!(grouped_params.w1, explicit_params.w1, 3.0e-3);
    compare_grouped_grad!(grouped_params.b1, explicit_params.b1, 3.0e-3);
    compare_grouped_grad!(grouped_params.w2, explicit_params.w2, 3.0e-3);
    assert!(grouped_params.b2.grad_remove(&mut grouped_grads).is_none());
    assert!(explicit_params.b2.grad_remove(&mut explicit_grads).is_none());
}

#[test]
fn dense_controller_row_gradient_through_rollout_matches_finite_difference() {
    let device = BurnDevice::default();
    let npa = NpaConfig::growing_2d();
    let model = NpaModel::upstream_seeded(npa.clone(), 0x4a11);
    let layout = NpaParameterRowLayout2d::new(&npa);
    let row_count = layout.row_count();
    let row_dims = layout.max_row_dims();
    let mut rows = vec![0.0; row_count * row_dims];
    for (index, value) in rows.iter_mut().enumerate() {
        *value = (index as f32 * 0.017).sin() * 0.002;
    }
    let coordinate = (npa.hidden_dims + 2) * row_dims + npa.hidden_dims;
    let x_values = vec![-0.3, -0.2, 0.2, -0.1, -0.1, 0.3, 0.3, 0.2];
    let s_values = (0..4 * npa.state_dims)
        .map(|index| (index as f32 * 0.031).cos() * 0.07)
        .collect::<Vec<_>>();
    let targets = vec![test_burn_target(&device, 1.0, 0.1)];
    let config = test_direct_config(4);

    let tracked_rows = tensor3(rows.clone(), [1, row_count, row_dims], &device).require_grad();
    let adapter = BurnAdapterBatch::from_dense_residual_rows(tracked_rows.clone(), &npa);
    let params = BurnBaseParams::from_model(&model, &device).unwrap();
    let (x, s, _) = rollout_batch_chunk(
        &params,
        &adapter,
        &targets,
        &[0],
        tensor3(x_values.clone(), [1, 4, 2], &device),
        tensor3(s_values.clone(), [1, 4, npa.state_dims], &device),
        config,
        4,
        &mut StdRng::seed_from_u64(7),
        2,
        Tensor::<BurnBackend, 1>::zeros([1], &device),
        None,
    );
    let objective = x.sum() + s.sum().mul_scalar(0.1);
    let mut gradients = objective.backward();
    let analytic = tensor3_vec(
        tracked_rows
            .grad_remove(&mut gradients)
            .expect("controller-row rollout gradient"),
    )
    .unwrap()[coordinate];

    let evaluate = |values: Vec<f32>| {
        let adapter = BurnAdapterBatch::from_dense_residual_rows(
            tensor3(values, [1, row_count, row_dims], &device),
            &npa,
        );
        let params = BurnBaseParams::from_model(&model, &device).unwrap();
        let (x, s, _) = rollout_batch_chunk(
            &params,
            &adapter,
            &targets,
            &[0],
            tensor3(x_values.clone(), [1, 4, 2], &device),
            tensor3(s_values.clone(), [1, 4, npa.state_dims], &device),
            config,
            4,
            &mut StdRng::seed_from_u64(7),
            2,
            Tensor::<BurnBackend, 1>::zeros([1], &device),
            None,
        );
        (x.sum() + s.sum().mul_scalar(0.1)).inner().into_scalar()
    };
    let epsilon = 1.0e-3;
    let mut plus = rows.clone();
    plus[coordinate] += epsilon;
    let mut minus = rows;
    minus[coordinate] -= epsilon;
    let finite = (evaluate(plus) - evaluate(minus)) / (2.0 * epsilon);
    let tolerance = 2.0e-3_f32.max(finite.abs() * 2.0e-2);
    assert!(
        (analytic - finite).abs() <= tolerance,
        "controller-row rollout gradient mismatch: analytic={analytic:e} finite={finite:e} tolerance={tolerance:e}"
    );
}

#[test]
fn target2d_batched_loss_and_gradients_match_unbatched_mean() {
    let device = BurnDevice::default();
    let npa_config = NpaConfig::growing_2d();
    let model = NpaModel::upstream_seeded(npa_config.clone(), 31);
    let targets = vec![
        test_burn_target(&device, 1.0, 0.1),
        test_burn_target(&device, 1.0, 0.2),
    ];
    let adapters = (0..2)
        .map(|_| {
            BurnAdapterParams::from_adapter(
                &NpaLowRankAdapter::zeros(&npa_config, 1, 1.0),
                &model,
                &device,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let adapter_batch = BurnAdapterBatch::from_indices(&adapters, &[0, 1]);
    let mut config = test_direct_config(4);
    config.loss_config.image_size = 1;
    let x_values = vec![
        -0.30, -0.20, 0.15, -0.10, -0.05, 0.25, 0.30, 0.20, -0.25, 0.30, 0.20, 0.15, -0.10, -0.20,
        0.35, -0.05,
    ];
    let s_values = (0..2 * 4 * npa_config.state_dims)
        .map(|idx| (idx as f32 * 0.013).sin() * 0.25)
        .collect::<Vec<_>>();
    let x_batch = tensor3(x_values.clone(), [2, 4, 2], &device).require_grad();
    let s_batch = tensor3(s_values.clone(), [2, 4, npa_config.state_dims], &device).require_grad();
    let batch_loss = target_splat_loss_batch(
        &x_batch,
        &s_batch,
        &targets,
        &[0, 1],
        config,
        &adapter_batch,
        Tensor::<BurnBackend, 1>::zeros([2], &device),
    )
    .unwrap();
    let batch_total = loss_scalars(&batch_loss).unwrap().total;
    let mut batch_grads = batch_loss.total.backward();
    let batch_x_grad = tensor3_vec(
        x_batch
            .grad_remove(&mut batch_grads)
            .expect("batched position gradient"),
    )
    .unwrap();
    let batch_s_grad = tensor3_vec(
        s_batch
            .grad_remove(&mut batch_grads)
            .expect("batched state gradient"),
    )
    .unwrap();

    let mut unbatched_total = 0.0_f32;
    let mut expected_x_grad = Vec::with_capacity(batch_x_grad.len());
    let mut expected_s_grad = Vec::with_capacity(batch_s_grad.len());
    for batch in 0..2 {
        let x_start = batch * 4 * 2;
        let s_start = batch * 4 * npa_config.state_dims;
        let x = tracked_tensor(x_values[x_start..x_start + 4 * 2].to_vec(), [4, 2], &device);
        let s = tracked_tensor(
            s_values[s_start..s_start + 4 * npa_config.state_dims].to_vec(),
            [4, npa_config.state_dims],
            &device,
        );
        let loss = target_splat_loss(
            &x,
            &s,
            &targets[batch],
            config,
            &adapters[batch],
            Tensor::<BurnBackend, 1>::zeros([1], &device),
        );
        unbatched_total += loss_scalars(&loss).unwrap().total;
        let mut grads = loss.total.backward();
        expected_x_grad.extend(
            tensor_vec(x.grad_remove(&mut grads).expect("position gradient"))
                .unwrap()
                .into_iter()
                .map(|value| value * 0.5),
        );
        expected_s_grad.extend(
            tensor_vec(s.grad_remove(&mut grads).expect("state gradient"))
                .unwrap()
                .into_iter()
                .map(|value| value * 0.5),
        );
    }

    assert!((batch_total - unbatched_total * 0.5).abs() < 1.0e-5);
    assert!(max_abs_difference(&batch_x_grad, &expected_x_grad) < 1.0e-5);
    assert!(max_abs_difference(&batch_s_grad, &expected_s_grad) < 1.0e-5);
}

#[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
#[test]
fn target2d_training_loss_routes_auto_to_device_adjoint() {
    let device = BurnDevice::default();
    let npa_config = NpaConfig::growing_2d();
    let model = NpaModel::upstream_seeded(npa_config.clone(), 37);
    let targets = vec![test_burn_target(&device, 1.0, 0.1)];
    let adapters = vec![
        BurnAdapterParams::from_adapter(
            &NpaLowRankAdapter::zeros(&npa_config, 1, 1.0),
            &model,
            &device,
        )
        .unwrap(),
    ];
    let adapter_batch = BurnAdapterBatch::from_indices(&adapters, &[0]);
    let mut config = test_direct_config(128);
    config.loss_config.image_size = 1;
    config.target2d_loss_backend = Target2dLossBackend::Auto;
    let device_hits_before = TARGET2D_CUBE_ADJOINT_DEVICE_HITS.load(Ordering::Relaxed);

    let loss = target_splat_loss_batch(
        &Tensor::<BurnBackend, 3>::zeros([1, 128, 2], &device).require_grad(),
        &Tensor::<BurnBackend, 3>::zeros([1, 128, npa_config.state_dims], &device).require_grad(),
        &targets,
        &[0],
        config,
        &adapter_batch,
        Tensor::<BurnBackend, 1>::zeros([1], &device),
    )
    .unwrap();
    let scalar = loss.total.inner().into_scalar();

    assert!(scalar.is_finite());
    assert!(
        TARGET2D_CUBE_ADJOINT_DEVICE_HITS.load(Ordering::Relaxed) > device_hits_before,
        "automatic Target2D routing did not invoke the device adjoint"
    );
}

#[test]
fn direct_particle_pool_persists_state_on_backend() {
    let device = BurnDevice::default();
    let config = test_direct_config(4);
    let mut pool = BurnDeviceParticlePool::new(2, 4, 16, 0.1, config, &device);
    pool.update_batch_with_ages(
        &[1],
        &[37],
        Tensor::<BurnBackend, 3>::full([1, 4, 2], 0.25, &device),
        Tensor::<BurnBackend, 3>::full([1, 4, 16], 0.5, &device),
    )
    .unwrap();

    let inner_device = pool.positions.device();
    let index = inner_index_tensor(&[1], &inner_device);
    assert!(
        tensor3_vec(pool.positions.clone().select(0, index.clone()))
            .unwrap()
            .iter()
            .all(|value| (*value - 0.25).abs() < 1.0e-6)
    );
    assert!(
        tensor3_vec(pool.states.clone().select(0, index))
            .unwrap()
            .iter()
            .all(|value| (*value - 0.5).abs() < 1.0e-6)
    );
    assert_eq!(pool.ages, vec![0, 37]);
    let snapshot = pool.target2d_snapshot().unwrap();
    let mut restored = BurnDeviceParticlePool::new(2, 4, 16, 0.1, config, &device);
    restored
        .restore_target2d_snapshot(&snapshot, &device)
        .unwrap();
    assert_eq!(restored.ages, vec![0, 37]);
    restored.reset();
    assert_eq!(restored.ages, vec![0, 0]);
}

#[test]
fn direct_particle_pool_recycles_selected_rows_at_the_age_limit() {
    let device = BurnDevice::default();
    let config = test_direct_config(4);
    let mut pool = BurnDeviceParticlePool::new(1, 4, 16, 0.1, config, &device);
    pool.update_batch_with_ages(
        &[0],
        &[37],
        Tensor::<BurnBackend, 3>::full([1, 4, 2], 0.25, &device),
        Tensor::<BurnBackend, 3>::full([1, 4, 16], 0.5, &device),
    )
    .unwrap();

    let mut rng = StdRng::seed_from_u64(11);
    let batch = pool
        .sample_batch_with_fresh_rows(
            &mut rng,
            1,
            BurnPoolSampling {
                fresh_seed_rows: 0,
                max_age_steps: Some(32),
                age_strata: 0,
                event_preference: None,
            },
            config,
            &device,
        )
        .unwrap();
    assert_eq!(batch.ages, vec![0]);
    assert!(
        tensor3_vec(batch.x.inner())
            .unwrap()
            .iter()
            .any(|value| (*value - 0.25).abs() > 1.0e-6)
    );
    assert!(
        tensor3_vec(batch.s.inner())
            .unwrap()
            .iter()
            .any(|value| (*value - 0.5).abs() > 1.0e-6)
    );
}

#[test]
fn age_stratified_pool_sampling_covers_available_bands_without_duplicates() {
    let ages = (0..16).map(|index| index * 64).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(17);
    let indices = sample_pool_indices_by_age(&mut rng, &ages, 8, 0, Some(1024), 4);

    assert_eq!(indices.len(), 8);
    let unique = indices.iter().copied().collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), indices.len());
    let mut counts = [0usize; 4];
    for index in indices {
        counts[pool_age_stratum(ages[index], 1024, 4)] += 1;
    }
    assert_eq!(counts, [2, 2, 2, 2]);
}

#[test]
fn age_stratified_pool_sampling_preserves_explicit_fresh_slots() {
    let ages = (0..16).map(|index| index * 64).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(23);
    let indices = sample_pool_indices_by_age(&mut rng, &ages, 8, 1, Some(1024), 4);

    assert_eq!(indices.len(), 8);
    let unique = indices.iter().copied().collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), indices.len());
    let persistent_counts = indices[1..]
        .iter()
        .map(|index| pool_age_stratum(ages[*index], 1024, 4))
        .fold([0usize; 4], |mut counts, stratum| {
            counts[stratum] += 1;
            counts
        });
    assert!(persistent_counts.iter().all(|count| *count >= 1));
}

#[test]
fn event_aware_pool_sampling_prefers_rows_that_cross_the_next_event() {
    let ages = vec![0, 16, 32, 48, 56, 64, 96, 112];
    let preference = BurnPoolEventPreference {
        start_step: 64,
        end_step: 0,
        interval_steps: 64,
        lookahead_steps: 16,
        min_rows: 3,
    };
    let mut rng = StdRng::seed_from_u64(29);
    let indices = sample_pool_indices_by_age_with_event_preference(
        &mut rng,
        &ages,
        4,
        0,
        None,
        0,
        Some(preference),
    );

    assert_eq!(indices.len(), 4);
    assert_eq!(
        indices
            .iter()
            .filter(|index| pool_age_crosses_preferred_event(ages[**index], preference))
            .count(),
        3
    );
}

#[test]
fn event_aware_pool_sampling_restores_seed_rows_when_event_ages_are_absent() {
    let device = BurnDevice::default();
    let config = test_direct_config(4);
    let mut pool = BurnDeviceParticlePool::new(8, 4, 16, 0.1, config, &device);
    pool.ages.fill(700);
    let mut rng = StdRng::seed_from_u64(31);
    let batch = pool
        .sample_batch_with_fresh_rows(
            &mut rng,
            4,
            BurnPoolSampling {
                fresh_seed_rows: 0,
                max_age_steps: Some(1024),
                age_strata: 4,
                event_preference: Some(BurnPoolEventPreference {
                    start_step: 64,
                    end_step: 512,
                    interval_steps: 64,
                    lookahead_steps: 96,
                    min_rows: 3,
                }),
            },
            config,
            &device,
        )
        .unwrap();

    assert_eq!(batch.ages.iter().filter(|age| **age == 0).count(), 3);
    assert_eq!(batch.ages.iter().filter(|age| **age == 700).count(), 1);
}

#[test]
fn direct_particle_pool_restores_unhealthy_trajectory_rows() {
    let device = BurnDevice::default();
    let config = test_direct_config(4);
    let pool = BurnDeviceParticlePool::new(2, 4, 16, 0.1, config, &device);
    let mut positions = vec![0.25; 2 * 4 * 2];
    positions[4 * 2] = f32::NAN;
    let mut states = vec![0.5; 2 * 4 * 16];
    states[4 * 16] = 100.0;
    let (restored_x, restored_s, unhealthy) = pool
        .restore_unhealthy_batch(
            &[0, 1],
            tensor3(positions, [2, 4, 2], &device),
            tensor3(states, [2, 4, 16], &device),
        )
        .unwrap();

    assert_eq!(
        unhealthy.into_data().to_vec::<u32>().unwrap(),
        vec![0, 1]
    );
    let restored_x = tensor3_vec(restored_x.inner()).unwrap();
    let restored_s = tensor3_vec(restored_s.inner()).unwrap();
    assert!(restored_x[..4 * 2]
        .iter()
        .all(|value| (*value - 0.25).abs() < 1.0e-6));
    assert!(restored_s[..4 * 16]
        .iter()
        .all(|value| (*value - 0.5).abs() < 1.0e-6));

    let inner_device = pool.positions.device();
    let second = inner_index_tensor(&[1], &inner_device);
    assert_eq!(
        &restored_x[4 * 2..],
        tensor3_vec(pool.initial_positions.clone().select(0, second.clone()))
            .unwrap()
            .as_slice()
    );
    assert_eq!(
        &restored_s[4 * 16..],
        tensor3_vec(pool.initial_states.clone().select(0, second))
            .unwrap()
            .as_slice()
    );
}

#[test]
fn hyper_e2e_device_particle_pool_is_bounded_and_persistent() {
    let device = BurnDevice::default();
    let particle_count = 4;
    let state_dims = 16;
    let mut pool = BurnE2eDeviceParticlePool::new(2, particle_count, state_dims, 2, &device);
    let mut rng = StdRng::seed_from_u64(7);
    let config = test_direct_config(particle_count);
    let first = pool
        .sample_batch(&[10, 10], &mut rng, &[], 0.1, config, &device)
        .unwrap();
    assert_eq!(first.slots.len(), 2);
    assert_ne!(first.slots[0], first.slots[1]);
    pool.update_batch(
        &first.slots,
        Tensor::<BurnBackend, 3>::full([2, particle_count, 2], 0.25, &device),
        Tensor::<BurnBackend, 3>::full([2, particle_count, state_dims], 0.5, &device),
    )
    .unwrap();
    let persisted = pool
        .sample_batch(&[10, 10], &mut rng, &[], 0.1, config, &device)
        .unwrap();
    assert!(
        tensor3_vec(persisted.x.inner())
            .unwrap()
            .iter()
            .all(|value| (*value - 0.25).abs() < 1.0e-6)
    );
    assert!(
        tensor3_vec(persisted.s.inner())
            .unwrap()
            .iter()
            .all(|value| (*value - 0.5).abs() < 1.0e-6)
    );
    let refreshed = pool
        .sample_batch(&[10, 10], &mut rng, &[0, 1], 0.1, config, &device)
        .unwrap();
    assert_eq!(refreshed.seed_replacements, 2);
    assert!(
        tensor3_vec(refreshed.x.inner())
            .unwrap()
            .iter()
            .any(|value| (*value - 0.25).abs() > 1.0e-3)
    );
    pool.update_batch(
        &first.slots,
        Tensor::<BurnBackend, 3>::full([2, particle_count, 2], 2.0, &device),
        Tensor::<BurnBackend, 3>::full([2, particle_count, state_dims], -3.0, &device),
    )
    .unwrap();
    let clamped = pool
        .sample_batch(&[10], &mut rng, &[], 0.1, config, &device)
        .unwrap();
    assert!(
        tensor3_vec(clamped.x.inner())
            .unwrap()
            .iter()
            .all(|value| (*value - 1.0).abs() < 1.0e-6)
    );
    assert!(
        tensor3_vec(clamped.s.inner())
            .unwrap()
            .iter()
            .all(|value| (*value + 3.0).abs() < 1.0e-6)
    );
    pool.sample_batch(&[12], &mut rng, &[], 0.1, config, &device)
        .unwrap();
    assert_eq!(pool.example_slots.len(), 2);
    assert!(pool.example_slots.keys().any(|(example, _)| *example == 12));
}

#[test]
fn hyper_e2e_particle_pool_repetition_reset_evicts_only_selected_identity() {
    let device = BurnDevice::default();
    let particle_count = 4;
    let mut pool = BurnE2eDeviceParticlePool::new(4, particle_count, 16, 2, &device);
    let mut rng = StdRng::seed_from_u64(11);
    let config = test_direct_config(particle_count);
    let first = pool
        .sample_batch(&[3, 3, 4, 4], &mut rng, &[], 0.1, config, &device)
        .unwrap();
    pool.update_batch(
        &first.slots,
        Tensor::<BurnBackend, 3>::full([4, particle_count, 2], 0.25, &device),
        Tensor::<BurnBackend, 3>::full([4, particle_count, 16], 0.5, &device),
    )
    .unwrap();

    pool.reset_examples(&[3]);
    assert!(!pool.example_slots.keys().any(|(example, _)| *example == 3));
    assert!(pool.example_slots.keys().any(|(example, _)| *example == 4));
    let reset = pool
        .sample_batch(&[3, 3], &mut rng, &[], 0.1, config, &device)
        .unwrap();
    assert!(
        tensor3_vec(reset.x.inner())
            .unwrap()
            .iter()
            .any(|value| (*value - 0.25).abs() > 1.0e-3)
    );
}

#[test]
fn hyper_e2e_device_particle_pool_sanitizes_nonfinite_updates() {
    let device = BurnDevice::default();
    let particle_count = 2;
    let state_dims = 16;
    let mut pool = BurnE2eDeviceParticlePool::new(1, particle_count, state_dims, 1, &device);
    let mut rng = StdRng::seed_from_u64(19);
    let config = test_direct_config(particle_count);
    let initial = pool
        .sample_batch(&[7], &mut rng, &[], 0.1, config, &device)
        .unwrap();
    pool.update_batch(
        &initial.slots,
        tensor3(
            vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.25],
            [1, particle_count, 2],
            &device,
        ),
        tensor3(
            std::iter::repeat_n([f32::NAN, f32::INFINITY], particle_count * state_dims / 2)
                .flatten()
                .collect(),
            [1, particle_count, state_dims],
            &device,
        ),
    )
    .unwrap();

    let persisted = pool
        .sample_batch(&[7], &mut rng, &[], 0.1, config, &device)
        .unwrap();
    let positions = tensor3_vec(persisted.x.inner()).unwrap();
    let states = tensor3_vec(persisted.s.inner()).unwrap();
    assert_eq!(positions, [0.0, 0.0, 0.0, 0.25]);
    assert!(states.iter().all(|value| *value == 0.0));
}

#[test]
fn hyper_e2e_particle_pool_erases_only_a_local_state_neighborhood() {
    let device = BurnDevice::default();
    let particle_count = 4;
    let state_dims = 16;
    let mut pool = BurnE2eDeviceParticlePool::new(1, particle_count, state_dims, 1, &device);
    let mut rng = StdRng::seed_from_u64(31);
    let mut config = test_direct_config(particle_count);
    let initial = pool
        .sample_batch(&[7], &mut rng, &[], 0.1, config, &device)
        .unwrap();
    pool.update_batch(
        &initial.slots,
        tensor3(
            vec![-0.9, 0.0, -0.3, 0.0, 0.3, 0.0, 0.9, 0.0],
            [1, particle_count, 2],
            &device,
        ),
        Tensor::<BurnBackend, 3>::ones([1, particle_count, state_dims], &device),
    )
    .unwrap();
    config.brush_size = 0.05;
    let damaged = pool
        .sample_batch(&[7], &mut rng, &[], 0.1, config, &device)
        .unwrap();
    let states = tensor3_vec(damaged.s.inner()).unwrap();
    assert_eq!(
        states.iter().filter(|value| **value == 0.0).count(),
        state_dims
    );
    assert_eq!(
        states.iter().filter(|value| **value == 1.0).count(),
        3 * state_dims
    );
}

#[test]
fn hyper_e2e_seed_replacement_cadence_is_per_identity_trajectory() {
    let mut counts = vec![0usize; 2];
    let identities = [0usize, 0, 1, 1];
    assert!(per_identity_seed_replacement_rows(&identities, &mut counts, 4).is_empty());
    assert_eq!(counts, [2, 2]);
    assert_eq!(
        per_identity_seed_replacement_rows(&identities, &mut counts, 4),
        [1, 3]
    );
    assert_eq!(counts, [0, 0]);
}

#[test]
fn hyper_e2e_upstream_repetition_resets_follow_identity_optimizer_steps() {
    let identities = [0usize, 0, 1, 1, 2, 2];
    let steps = [10_000usize, 9_999, 20_000];
    assert_eq!(
        upstream_growing_repetition_reset_identities(&identities, &steps),
        [0, 2]
    );
}

#[test]
fn hyper_e2e_scheduled_seed_replacements_cover_distinct_batch_rows() {
    let identities = [0usize, 0, 1, 1];
    let mut counts = vec![0usize; 2];
    assert_eq!(
        e2e_seed_replacement_rows(&identities, &mut counts, 128, 3, 4, 2),
        Vec::<usize>::new()
    );
    assert_eq!(
        e2e_seed_replacement_rows(&identities, &mut counts, 128, 4, 4, 2),
        [2, 3]
    );
    assert_eq!(
        e2e_seed_replacement_rows(&identities, &mut counts, 128, 8, 4, 2),
        [0, 1]
    );
}

#[test]
fn hyper_e2e_seed_replacement_sources_are_deduplicated() {
    let identities = [0usize, 0, 1, 1];
    let mut counts = vec![1usize, 1];
    assert_eq!(
        e2e_seed_replacement_rows(&identities, &mut counts, 2, 4, 4, 2),
        [0, 1, 2, 3]
    );
    assert_eq!(counts, [1, 1]);
}

#[test]
fn dense_perception_matches_reference_kernel_fixture() {
    let npa_config = NpaConfig::growing_2d();
    let grid = burn_automata_kernels::HashGridConfig::growing_2d();
    let (positions, states) = seed_particles_scaled(
        1,
        4,
        npa_config.state_dims,
        npa_config.spatial_dims,
        17,
        crate::ParticleSeed::UniformCircle,
        0.08,
    );
    let options = burn_automata_kernels::PerceptionOptions {
        state_grad: npa_config.state_grad,
        density_grad: npa_config.density_grad,
        eps0: npa_config.eps0,
        scale_equivariance: npa_config.scale_equivariant(),
        particle_density_equivariance: npa_config.particle_density_equivariant(),
        log_norm_grad: npa_config.log_norm_grad,
        log_norm_density_grad: npa_config.log_norm_density_grad,
        hybrid_state_gradient: true,
        position_features: npa_config.position_features,
    };
    let reference = burn_automata_kernels::perceive_with_options(
        &positions,
        &states,
        1,
        4,
        npa_config.state_dims,
        &grid,
        options,
    )
    .unwrap();
    let device = BurnDevice::default();
    let x = tensor(
        positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect(),
        [4, 2],
        &device,
    );
    let s = tensor(states, [4, npa_config.state_dims], &device);
    let config = DirectBasisTrainConfig {
        steps: 0,
        report_interval: 1,
        example_batch_size: 1,
        tbptt_chunk_steps: 1,
        loss_on_final_chunk_only: false,
        use_particle_pool: false,
        pool_size: 0,
        inject_seed_interval: 0,
        brush_size: 0.0,
        stopgrad_pos: npa_config.stopgrad_pos,
        stopgrad_state: npa_config.stopgrad_state,
        rollout_particles: 4,
        rollout_step_min: 1,
        rollout_steps: 1,
        update_prob: 1.0,
        seed: 17,
        seed_scale: 0.08,
        seed_mode: crate::ParticleSeed::UniformCircle,
        grid_eps: grid.eps,
        motion_scale: npa_config.alpha * npa_config.motion_eps(grid.eps),
        loss_config: crate::Target2dLossConfig::default(),
        target2d_loss_backend: Target2dLossBackend::Dense,
        perception_backend: PerceptionRolloutBackend::Dense,
        per_parameter_grad_normalization: false,
        base_sgd: SgdConfig {
            learning_rate: 0.0,
            weight_decay: 0.0,
            grad_clip_norm: 0.0,
        },
        adapter_sgd: SgdConfig {
            learning_rate: 0.0,
            weight_decay: 0.0,
            grad_clip_norm: 0.0,
        },
        adapter_l2_weight: 0.0,
        update_base: true,
        eval_examples: 1,
        eval_interval: 0,
        eval_batch_size: 1,
        eval_seed: 17,
        system_memory_budget_gb: None,
        gpu_memory_budget_gb: None,
        max_dense_train_particles: 4,
        max_dense_chunk_floats: 1_000_000,
        max_splat_chunk_floats: 1_000_000,
    };
    let features = tensor_vec(dense_perception(&x, &s, config).inner()).unwrap();
    let max_abs_diff = features
        .iter()
        .zip(reference.features.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_abs_diff < 2.0e-3,
        "dense Burn perception diverged from reference: max_abs_diff={max_abs_diff}"
    );
}

#[test]
fn perception_tiled_adjoint_matches_dense_vjp_fixture() {
    let npa_config = NpaConfig::growing_2d();
    let grid = burn_automata_kernels::HashGridConfig::growing_2d();
    let particle_count = 16;
    let (positions, states) = seed_particles_scaled(
        1,
        particle_count,
        npa_config.state_dims,
        npa_config.spatial_dims,
        23,
        crate::ParticleSeed::UniformCircle,
        0.05,
    );
    let device = BurnDevice::default();
    let config = DirectBasisTrainConfig {
        steps: 0,
        report_interval: 1,
        example_batch_size: 1,
        tbptt_chunk_steps: 1,
        loss_on_final_chunk_only: false,
        use_particle_pool: false,
        pool_size: 0,
        inject_seed_interval: 0,
        brush_size: 0.0,
        stopgrad_pos: false,
        stopgrad_state: false,
        rollout_particles: particle_count,
        rollout_step_min: 1,
        rollout_steps: 1,
        update_prob: 1.0,
        seed: 23,
        seed_scale: 0.05,
        seed_mode: crate::ParticleSeed::UniformCircle,
        grid_eps: grid.eps,
        motion_scale: npa_config.alpha * npa_config.motion_eps(grid.eps),
        loss_config: crate::Target2dLossConfig::default(),
        target2d_loss_backend: Target2dLossBackend::Dense,
        perception_backend: PerceptionRolloutBackend::Dense,
        per_parameter_grad_normalization: false,
        base_sgd: SgdConfig {
            learning_rate: 0.0,
            weight_decay: 0.0,
            grad_clip_norm: 0.0,
        },
        adapter_sgd: SgdConfig {
            learning_rate: 0.0,
            weight_decay: 0.0,
            grad_clip_norm: 0.0,
        },
        adapter_l2_weight: 0.0,
        update_base: true,
        eval_examples: 1,
        eval_interval: 0,
        eval_batch_size: 1,
        eval_seed: 23,
        system_memory_budget_gb: None,
        gpu_memory_budget_gb: None,
        max_dense_train_particles: particle_count,
        max_dense_chunk_floats: 1_000_000,
        max_splat_chunk_floats: 1_000_000,
    };
    let position_values = positions
        .iter()
        .flat_map(|position| [position[0], position[1]])
        .collect::<Vec<_>>();
    let reference_positions = positions.clone();
    let reference_states = states.clone();
    let x_dense = tensor3(position_values.clone(), [1, particle_count, 2], &device).require_grad();
    let s_dense = tensor3(
        states.clone(),
        [1, particle_count, npa_config.state_dims],
        &device,
    )
    .require_grad();
    let dense_features = dense_perception_batch(&x_dense, &s_dense, config);
    let feature_dims = dense_features.shape().dims::<3>()[2];
    let feature_weights = (0..feature_dims * particle_count)
        .map(|idx| (((idx * 17) % 13) as f32 - 6.0) * 0.01)
        .collect::<Vec<_>>();
    let reference_feature_weights = feature_weights.clone();
    let weights = tensor3(feature_weights, [1, particle_count, feature_dims], &device);
    let dense_loss = dense_features.clone().mul(weights.clone()).sum();
    let dense_values = tensor3_vec(dense_features.inner()).unwrap();
    let mut dense_grads = dense_loss.backward();
    let dense_x_grad = tensor3_vec(
        x_dense
            .grad_remove(&mut dense_grads)
            .unwrap_or_else(|| x_dense.clone().inner().zeros_like()),
    )
    .unwrap();
    let dense_s_grad = tensor3_vec(
        s_dense
            .grad_remove(&mut dense_grads)
            .unwrap_or_else(|| s_dense.clone().inner().zeros_like()),
    )
    .unwrap();

    let x_tiled = tensor3(position_values, [1, particle_count, 2], &device).require_grad();
    let s_tiled =
        tensor3(states, [1, particle_count, npa_config.state_dims], &device).require_grad();
    let tiled_features = perception_tiled_adjoint_batch(x_tiled.clone(), s_tiled.clone(), config);
    let tiled_loss = tiled_features.clone().mul(weights).sum();
    let tiled_values = tensor3_vec(tiled_features.inner()).unwrap();
    let mut tiled_grads = tiled_loss.backward();
    let tiled_x_grad = tensor3_vec(
        x_tiled
            .grad_remove(&mut tiled_grads)
            .unwrap_or_else(|| x_tiled.clone().inner().zeros_like()),
    )
    .unwrap();
    let tiled_s_grad = tensor3_vec(
        s_tiled
            .grad_remove(&mut tiled_grads)
            .unwrap_or_else(|| s_tiled.clone().inner().zeros_like()),
    )
    .unwrap();

    let feature_diff = max_abs_difference(&dense_values, &tiled_values);
    let (position_grad_idx, position_grad_diff, position_dense, position_tiled) =
        max_abs_difference_with_index(&dense_x_grad, &tiled_x_grad);
    let (state_grad_idx, state_grad_diff, state_dense, state_tiled) =
        max_abs_difference_with_index(&dense_s_grad, &tiled_s_grad);
    let manual_adjoint = burn_automata_kernels::perceive_adjoint_with_options(
        &reference_positions,
        &reference_states,
        1,
        particle_count,
        npa_config.state_dims,
        &perception_reference_grid(grid.eps),
        perception_reference_options(grid.eps),
        &reference_feature_weights,
    )
    .unwrap();
    let manual_state = manual_adjoint.state[state_grad_idx];
    let state_grad_relative = state_grad_diff / state_dense.abs().max(state_tiled.abs()).max(1.0);
    let position_grad_relative =
        position_grad_diff / position_dense.abs().max(position_tiled.abs()).max(1.0);
    assert!(
        feature_diff < 2.0e-3,
        "tiled perception features diverged from dense Burn features: max_abs_diff={feature_diff}"
    );
    assert!(
        position_grad_diff < 2.0e-1
            && (position_grad_diff < 2.0e-2 || position_grad_relative < 5.0e-3),
        "tiled perception position VJP diverged from dense Burn VJP: idx={position_grad_idx} dense={position_dense} tiled={position_tiled} max_abs_diff={position_grad_diff} rel_diff={position_grad_relative}"
    );
    assert!(
        state_grad_diff < 2.0e-2 && state_grad_relative < 5.0e-3,
        "tiled perception state VJP diverged from dense Burn VJP: idx={state_grad_idx} dense={state_dense} tiled={state_tiled} manual={manual_state} max_abs_diff={state_grad_diff} rel_diff={state_grad_relative}"
    );
}

#[test]
fn adaptive_perception_autodiff_matches_reference_state_vjp() {
    adaptive_perception_autodiff_matches_reference_state_vjp_for(
        burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible,
    );
}

#[test]
fn adaptive_perception_coarse_exposure_uses_measure_at_fixed_support() {
    let device = BurnDevice::default();
    let particle_count = 3;
    let positions = vec![0.0, 0.0, 0.04, 0.0, 0.8, 0.8];
    let reference_positions = positions
        .chunks_exact(2)
        .map(|position| [position[0], position[1], 0.0, 0.0])
        .collect::<Vec<_>>();
    let states = vec![0.0; particle_count * 2];
    let represented_measure = vec![1.0, 4.0, 1.0];
    let bandwidth = vec![0.1; particle_count];
    let config = burn_automata_kernels::AdaptivePerceptionConfig {
        graph_policy: burn_automata_kernels::AdaptiveGraphPolicy::RawSupport,
        max_neighbors: particle_count,
        reference_measure: 1.0,
        min_bandwidth: 0.1,
        max_bandwidth: 0.1,
        ..burn_automata_kernels::AdaptivePerceptionConfig::growing_2d()
    };
    let options = burn_automata_kernels::AdaptiveNpaPerceptionOptions {
        eps0: 0.1,
        scale_equivariance: true,
        particle_density_equivariance: true,
        log_norm_grad: true,
        log_norm_density_grad: true,
        position_features: false,
    };
    let output = adaptive_perception::adaptive_npa_perception_batch(
        tensor3(positions, [1, particle_count, 2], &device),
        tensor3(states.clone(), [1, particle_count, 2], &device),
        tensor(
            represented_measure.clone(),
            [1, particle_count],
            &device,
        ),
        tensor(bandwidth.clone(), [1, particle_count], &device),
        config,
        options,
        burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible,
    );
    let exposure = test_tensor_values(output.coarse_exposure.inner());
    let reference = burn_automata_kernels::adaptive_npa_perceive_all_pairs(
        &reference_positions,
        &states,
        &represented_measure,
        &bandwidth,
        1,
        particle_count,
        2,
        config,
        options,
    )
    .unwrap();
    assert!(exposure[0] > 0.0 && exposure[0] < 1.0);
    assert!(exposure[1] > exposure[0]);
    assert_eq!(exposure[2], 0.0);
    assert!(
        max_abs_difference(&exposure, &reference.coarse_exposure) <= 2.0e-5,
        "device coarse exposure diverged from the represented-measure oracle"
    );
}

#[test]
fn normalized_adaptive_perception_autodiff_matches_reference_state_vjp() {
    adaptive_perception_autodiff_matches_reference_state_vjp_for(
        burn_automata_kernels::AdaptivePerceptionSemantics::NormalizedAdaptive,
    );
}

fn adaptive_perception_autodiff_matches_reference_state_vjp_for(
    semantics: burn_automata_kernels::AdaptivePerceptionSemantics,
) {
    let batch_size = 2;
    let particle_count = 12;
    let state_dims = 4;
    let positions = (0..batch_size * particle_count)
        .flat_map(|index| {
            let batch = index / particle_count;
            let local = index % particle_count;
            let angle = local as f32 * 2.399_963_1 + batch as f32 * 0.17;
            let radius = 0.045 * (local as f32 + 1.0).sqrt();
            [radius * angle.cos(), radius * angle.sin()]
        })
        .collect::<Vec<_>>();
    let reference_positions = positions
        .chunks_exact(2)
        .map(|position| [position[0], position[1], 0.0, 0.0])
        .collect::<Vec<_>>();
    let states = (0..batch_size * particle_count * state_dims)
        .map(|index| (index as f32 * 0.173).sin())
        .collect::<Vec<_>>();
    let represented_measure = (0..batch_size * particle_count)
        .map(|index| 0.35 + (index % 5) as f32 * 0.11)
        .collect::<Vec<_>>();
    let bandwidth = (0..batch_size * particle_count)
        .map(|index| {
            if index % 3 == 0 {
                0.1
            } else {
                0.19 + (index % 2) as f32 * 0.025
            }
        })
        .collect::<Vec<_>>();
    let config = burn_automata_kernels::AdaptivePerceptionConfig {
        graph_policy: burn_automata_kernels::AdaptiveGraphPolicy::RawSupport,
        max_neighbors: particle_count,
        min_bandwidth: 0.1,
        max_bandwidth: 0.3,
        spacing_target_neighbors: 4.0,
        moment_condition_limit: 1.0e8,
        ..burn_automata_kernels::AdaptivePerceptionConfig::growing_2d()
    };
    let options = burn_automata_kernels::AdaptiveNpaPerceptionOptions {
        eps0: 0.1,
        scale_equivariance: true,
        particle_density_equivariance: true,
        log_norm_grad: true,
        log_norm_density_grad: true,
        position_features: false,
    };
    let feature_dims = config.feature_dims(state_dims);
    let feature_weights = (0..batch_size * particle_count * feature_dims)
        .map(|index| (index as f32 * 0.097).cos() * 0.1)
        .collect::<Vec<_>>();
    let device = BurnDevice::default();
    let state_tensor = tensor3(
        states.clone(),
        [batch_size, particle_count, state_dims],
        &device,
    )
    .require_grad();
    let perception_output = adaptive_perception::adaptive_npa_perception_batch(
        tensor3(
            positions,
            [batch_size, particle_count, 2],
            &device,
        ),
        state_tensor.clone(),
        tensor(
            represented_measure.clone(),
            [batch_size, particle_count],
            &device,
        ),
        tensor(
            bandwidth.clone(),
            [batch_size, particle_count],
            &device,
        ),
        config,
        options,
        semantics,
    );
    let coarse_exposure = perception_output
        .coarse_exposure
        .clone()
        .inner()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let features = perception_output.features;
    let feature_values = tensor3_vec(features.clone().inner()).unwrap();
    let objective = features
        .mul(tensor3(
            feature_weights.clone(),
            [batch_size, particle_count, feature_dims],
            &device,
        ))
        .sum();
    let mut gradients = objective.backward();
    let state_gradient =
        tensor3_vec(state_tensor.grad_remove(&mut gradients).unwrap()).unwrap();

    let (reference, reference_gradient) = match semantics {
        burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible => (
            burn_automata_kernels::adaptive_npa_perceive_all_pairs(
                &reference_positions,
                &states,
                &represented_measure,
                &bandwidth,
                batch_size,
                particle_count,
                state_dims,
                config,
                options,
            )
            .unwrap(),
            burn_automata_kernels::adaptive_npa_perceive_state_adjoint_all_pairs(
                &reference_positions,
                &states,
                &represented_measure,
                &bandwidth,
                batch_size,
                particle_count,
                state_dims,
                config,
                options,
                &feature_weights,
            )
            .unwrap(),
        ),
        burn_automata_kernels::AdaptivePerceptionSemantics::NormalizedAdaptive => (
            burn_automata_kernels::adaptive_perceive_all_pairs(
                &reference_positions,
                &states,
                &represented_measure,
                &bandwidth,
                batch_size,
                particle_count,
                state_dims,
                config,
            )
            .unwrap(),
            burn_automata_kernels::adaptive_perceive_state_adjoint_all_pairs(
                &reference_positions,
                &states,
                &represented_measure,
                &bandwidth,
                batch_size,
                particle_count,
                state_dims,
                config,
                &feature_weights,
            )
            .unwrap(),
        ),
    };
    assert!(
        max_abs_difference(&feature_values, &reference.features) <= 2.0e-5,
        "adaptive Burn forward diverged from the reference"
    );
    if semantics == burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible {
        assert!(
            max_abs_difference(&coarse_exposure, &reference.coarse_exposure) <= 2.0e-5,
            "adaptive Burn coarse exposure diverged from the reference"
        );
    }
    assert!(
        max_abs_difference(&state_gradient, &reference_gradient) <= 3.0e-5,
        "adaptive Burn state VJP diverged from the reference"
    );
}

#[test]
fn adaptive_sparse_perception_autodiff_matches_reference_state_vjp() {
    adaptive_sparse_perception_autodiff_matches_reference_state_vjp_for(
        burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible,
    );
}

#[test]
fn normalized_adaptive_sparse_perception_autodiff_matches_reference_state_vjp() {
    adaptive_sparse_perception_autodiff_matches_reference_state_vjp_for(
        burn_automata_kernels::AdaptivePerceptionSemantics::NormalizedAdaptive,
    );
}

fn adaptive_sparse_perception_autodiff_matches_reference_state_vjp_for(
    semantics: burn_automata_kernels::AdaptivePerceptionSemantics,
) {
    let batch_size = 1;
    let particle_count = 128;
    let state_dims = 4;
    let positions = (0..particle_count)
        .flat_map(|index| {
            let column = index % 16;
            let row = index / 16;
            let jitter = (index as f32 * 1.618_034).sin() * 0.003;
            [
                -0.675 + column as f32 * 0.09 + jitter,
                -0.63 + row as f32 * 0.18 - jitter,
            ]
        })
        .collect::<Vec<_>>();
    let reference_positions = positions
        .chunks_exact(2)
        .map(|position| [position[0], position[1], 0.0, 0.0])
        .collect::<Vec<_>>();
    let states = (0..particle_count * state_dims)
        .map(|index| (index as f32 * 0.173).sin())
        .collect::<Vec<_>>();
    let represented_measure = (0..particle_count)
        .map(|index| {
            if index % 7 == 0 {
                4.0 / 4096.0
            } else {
                1.0 / 4096.0
            }
        })
        .collect::<Vec<_>>();
    let bandwidth = (0..particle_count)
        .map(|index| if index % 7 == 0 { 0.2 } else { 0.1 })
        .collect::<Vec<_>>();
    let config = burn_automata_kernels::AdaptivePerceptionConfig {
        graph_policy: burn_automata_kernels::AdaptiveGraphPolicy::RawSupport,
        max_neighbors: particle_count,
        min_bandwidth: 0.1,
        max_bandwidth: 0.2,
        spacing_target_neighbors: 4.0,
        moment_condition_limit: 1.0e8,
        ..burn_automata_kernels::AdaptivePerceptionConfig::growing_2d()
    };
    let options = burn_automata_kernels::AdaptiveNpaPerceptionOptions {
        eps0: 0.1,
        scale_equivariance: true,
        particle_density_equivariance: true,
        log_norm_grad: true,
        log_norm_density_grad: true,
        position_features: false,
    };
    let feature_dims = config.feature_dims(state_dims);
    let feature_weights = (0..particle_count * feature_dims)
        .map(|index| (index as f32 * 0.097).cos() * 0.1)
        .collect::<Vec<_>>();
    let device = BurnDevice::default();
    let state_tensor =
        tensor3(states.clone(), [batch_size, particle_count, state_dims], &device).require_grad();
    let perception_output = adaptive_perception::adaptive_npa_perception_batch(
        tensor3(
            positions,
            [batch_size, particle_count, 2],
            &device,
        ),
        state_tensor.clone(),
        tensor(
            represented_measure.clone(),
            [batch_size, particle_count],
            &device,
        ),
        tensor(
            bandwidth.clone(),
            [batch_size, particle_count],
            &device,
        ),
        config,
        options,
        semantics,
    );
    let coarse_exposure = perception_output
        .coarse_exposure
        .inner()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let features = perception_output.features;
    let feature_values = tensor3_vec(features.clone().inner()).unwrap();
    let objective = features
        .mul(tensor3(
            feature_weights.clone(),
            [batch_size, particle_count, feature_dims],
            &device,
        ))
        .sum();
    let mut gradients = objective.backward();
    let state_gradient =
        tensor3_vec(state_tensor.grad_remove(&mut gradients).unwrap()).unwrap();

    let (reference, reference_gradient) = match semantics {
        burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible => (
            burn_automata_kernels::adaptive_npa_perceive_all_pairs(
                &reference_positions,
                &states,
                &represented_measure,
                &bandwidth,
                batch_size,
                particle_count,
                state_dims,
                config,
                options,
            )
            .unwrap(),
            burn_automata_kernels::adaptive_npa_perceive_state_adjoint_all_pairs(
                &reference_positions,
                &states,
                &represented_measure,
                &bandwidth,
                batch_size,
                particle_count,
                state_dims,
                config,
                options,
                &feature_weights,
            )
            .unwrap(),
        ),
        burn_automata_kernels::AdaptivePerceptionSemantics::NormalizedAdaptive => (
            burn_automata_kernels::adaptive_perceive_all_pairs(
                &reference_positions,
                &states,
                &represented_measure,
                &bandwidth,
                batch_size,
                particle_count,
                state_dims,
                config,
            )
            .unwrap(),
            burn_automata_kernels::adaptive_perceive_state_adjoint_all_pairs(
                &reference_positions,
                &states,
                &represented_measure,
                &bandwidth,
                batch_size,
                particle_count,
                state_dims,
                config,
                &feature_weights,
            )
            .unwrap(),
        ),
    };
    assert!(
        max_abs_difference(&feature_values, &reference.features) <= 2.0e-4,
        "adaptive sparse Burn forward diverged from the all-pairs reference"
    );
    if semantics == burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible {
        assert!(
            max_abs_difference(&coarse_exposure, &reference.coarse_exposure) <= 2.0e-4,
            "adaptive sparse Burn coarse exposure diverged from the all-pairs reference"
        );
    }
    assert!(
        max_abs_difference(&state_gradient, &reference_gradient) <= 3.0e-4,
        "adaptive sparse Burn state VJP diverged from the all-pairs reference"
    );
}

#[test]
fn adaptive_recurrent_state_gradient_matches_multistep_finite_difference() {
    let device = BurnDevice::default();
    let particle_count = 6;
    let steps = 4;
    let npa_config = NpaConfig::growing_2d();
    let model = NpaModel::upstream_seeded(npa_config.clone(), 0xa4da_710e);
    let positions = (0..particle_count)
        .flat_map(|particle| {
            let angle = particle as f32 * 2.399_963_1;
            let radius = 0.04 * (particle as f32 + 1.0).sqrt();
            [radius * angle.cos(), radius * angle.sin()]
        })
        .collect::<Vec<_>>();
    let states = (0..particle_count * npa_config.state_dims)
        .map(|index| (index as f32 * 0.071).sin() * 0.05)
        .collect::<Vec<_>>();
    let state_weights = (0..particle_count * npa_config.state_dims)
        .map(|index| (index as f32 * 0.113).cos())
        .collect::<Vec<_>>();
    let represented_measure = vec![1.0; particle_count];
    let bandwidth = vec![0.15, 0.18, 0.21, 0.16, 0.19, 0.22];
    let perception = burn_automata_kernels::AdaptivePerceptionConfig {
        graph_policy: burn_automata_kernels::AdaptiveGraphPolicy::RawSupport,
        max_neighbors: particle_count,
        min_bandwidth: 0.1,
        max_bandwidth: 0.3,
        spacing_target_neighbors: 4.0,
        moment_condition_limit: 1.0e8,
        ..burn_automata_kernels::AdaptivePerceptionConfig::growing_2d()
    };
    let perception_options = burn_automata_kernels::AdaptiveNpaPerceptionOptions {
        eps0: 0.1,
        scale_equivariance: true,
        particle_density_equivariance: true,
        log_norm_grad: true,
        log_norm_density_grad: true,
        position_features: false,
    };
    let mut config = test_direct_config(particle_count);
    config.motion_scale = 0.0;
    let targets = vec![test_burn_target(&device, 1.0, 0.1)];
    let objective_weights = tensor3(
        state_weights.clone(),
        [1, particle_count, npa_config.state_dims],
        &device,
    );

    let rollout_objective = |params: &BurnBaseBatch| {
        let mut rngs = [StdRng::seed_from_u64(17)];
        let (_, state, _, _) = rollout_adaptive_oracle_model_batch_chunk(
            params,
            None,
            &targets,
            &[0],
            tensor3(positions.clone(), [1, particle_count, 2], &device),
            tensor3(
                states.clone(),
                [1, particle_count, npa_config.state_dims],
                &device,
            ),
            tensor(
                represented_measure.clone(),
                [1, particle_count],
                &device,
            ),
            tensor(bandwidth.clone(), [1, particle_count], &device),
            Tensor::<BurnBackend, 2>::zeros([1, particle_count], &device),
            Tensor::<BurnBackend, 2>::ones([1, particle_count], &device),
            perception,
            perception_options,
            burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible,
            None,
            false,
            false,
            config,
            particle_count,
            &mut rngs,
            true,
            None,
            None,
            0,
            steps,
            Tensor::<BurnBackend, 1>::zeros([1], &device),
        );
        state.mul(objective_weights.clone()).sum()
    };

    let params = BurnBaseBatch::from_models(std::slice::from_ref(&model), &device).unwrap();
    let objective = rollout_objective(&params);
    let mut gradients = objective.backward();
    let analytic = tensor3_vec(
        params
            .w2
            .grad_remove(&mut gradients)
            .expect("adaptive recurrent parameter gradient"),
    )
    .unwrap();
    let (coordinate, analytic_value) = analytic
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, left), (_, right)| {
            left.abs()
                .partial_cmp(&right.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("finite adaptive recurrent gradient");
    assert!(
        analytic_value.abs() > 1.0e-6,
        "adaptive recurrent gradient fixture is degenerate"
    );

    let evaluate = |candidate: NpaModel| {
        let params =
            BurnBaseBatch::from_models(std::slice::from_ref(&candidate), &device).unwrap();
        rollout_objective(&params).inner().into_scalar()
    };
    let epsilon = 1.0e-3;
    let mut plus = model.clone();
    plus.weights.w2[coordinate] += epsilon;
    let mut minus = model;
    minus.weights.w2[coordinate] -= epsilon;
    let finite_difference = (evaluate(plus) - evaluate(minus)) / (2.0 * epsilon);
    let tolerance = 2.0e-3_f32.max(finite_difference.abs() * 5.0e-2);
    assert!(
        (analytic_value - finite_difference).abs() <= tolerance,
        "adaptive recurrent parameter gradient mismatch: coordinate={coordinate} analytic={analytic_value:e} finite={finite_difference:e} tolerance={tolerance:e}"
    );
}

#[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
#[test]
fn perception_cube_state_vjp_components_match_reference() {
    let npa_config = NpaConfig::growing_2d();
    let mut grid = burn_automata_kernels::HashGridConfig::growing_2d();
    grid.eps = 0.075;
    let particle_count = 16;
    let (positions, states) = seed_particles_scaled(
        1,
        particle_count,
        npa_config.state_dims,
        npa_config.spatial_dims,
        23,
        crate::ParticleSeed::UniformCircle,
        0.05,
    );
    let device = BurnDevice::default();
    let x = tensor3(
        positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect(),
        [1, particle_count, 2],
        &device,
    );
    let s = tensor3(
        states.clone(),
        [1, particle_count, npa_config.state_dims],
        &device,
    );

    for (hybrid_state_gradient, log_norm_grad) in
        [(false, false), (false, true), (true, false), (true, true)]
    {
        let mut options = perception_reference_options(grid.eps);
        options.hybrid_state_gradient = hybrid_state_gradient;
        options.log_norm_grad = log_norm_grad;
        let reference = burn_automata_kernels::perceive_with_options(
            &positions,
            &states,
            1,
            particle_count,
            npa_config.state_dims,
            &grid,
            options,
        )
        .unwrap();
        let feature_weights = (0..particle_count * reference.feature_dims)
            .map(|idx| {
                let feature = idx % reference.feature_dims;
                if (2 * npa_config.state_dims..4 * npa_config.state_dims).contains(&feature) {
                    (((idx * 17) % 13) as f32 - 6.0) * 0.01
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let reference_adjoint = burn_automata_kernels::perceive_adjoint_with_options(
            &positions,
            &states,
            1,
            particle_count,
            npa_config.state_dims,
            &grid,
            options,
            &feature_weights,
        )
        .unwrap();
        let mut cube_config = perception_cube_adjoint_config(grid.eps, false, true);
        cube_config.hybrid_state_gradient = hybrid_state_gradient;
        cube_config.log_norm_grad = log_norm_grad;
        let cube_forward = InnerBackend::perception_cube_forward(
            x.clone().inner(),
            s.clone().inner(),
            cube_config,
        )
        .expect("perception CubeCL backend")
        .expect("perception CubeCL forward");
        let cube_features = tensor3_vec(cube_forward.features).unwrap();
        let feature_diff = max_abs_difference(&reference.features, &cube_features);
        assert!(
            feature_diff < 2.0e-3,
            "CubeCL perception forward component mismatch: hybrid={hybrid_state_gradient} log_norm={log_norm_grad} max_abs_diff={feature_diff}"
        );
        let feature_grad = Tensor::<InnerBackend, 3>::from_data(
            TensorData::new(feature_weights, [1, particle_count, reference.feature_dims]),
            &x.clone().inner().device(),
        );
        let cube_adjoint = InnerBackend::perception_cube_adjoint(
            x.clone().inner(),
            s.clone().inner(),
            feature_grad,
            cube_config,
        )
        .expect("perception CubeCL backend")
        .expect("perception CubeCL state VJP");
        let cube_state = tensor3_vec(cube_adjoint.state_grad).unwrap();
        let (idx, max_abs_diff, reference_value, cube_value) =
            max_abs_difference_with_index(&reference_adjoint.state, &cube_state);
        let relative = max_abs_diff / reference_value.abs().max(cube_value.abs()).max(1.0);
        assert!(
            max_abs_diff < 2.0e-2 && relative < 5.0e-3,
            "CubeCL state VJP component mismatch: hybrid={hybrid_state_gradient} log_norm={log_norm_grad} idx={idx} reference={reference_value} cube={cube_value} max_abs_diff={max_abs_diff} relative={relative}"
        );
    }
}

#[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
#[test]
fn perception_sparse_cell_planes_match_reference_state_vjp() {
    let npa_config = NpaConfig::growing_2d();
    let grid = crate::upstream_growing_2d_hashgrid();
    let particle_count = 1024;
    let (positions, mut states) = seed_particles_scaled(
        1,
        particle_count,
        npa_config.state_dims,
        npa_config.spatial_dims,
        91,
        crate::ParticleSeed::UniformCircle,
        0.5,
    );
    for (index, state) in states.iter_mut().enumerate() {
        *state = (index as f32 * 0.013_731).sin() * 0.1;
    }
    let options = perception_reference_options(grid.eps);
    let reference = burn_automata_kernels::perceive_with_options(
        &positions,
        &states,
        1,
        particle_count,
        npa_config.state_dims,
        &grid,
        options,
    )
    .unwrap();
    let feature_dims = reference.feature_dims;
    let feature_weights = (0..particle_count * feature_dims)
        .map(|idx| (((idx * 17) % 19) as f32 - 9.0) * 1.0e-4)
        .collect::<Vec<_>>();
    let reference_adjoint = burn_automata_kernels::perceive_adjoint_with_options(
        &positions,
        &states,
        1,
        particle_count,
        npa_config.state_dims,
        &grid,
        options,
        &feature_weights,
    )
    .unwrap();

    let device = BurnDevice::default();
    let position_values = positions
        .iter()
        .flat_map(|position| [position[0], position[1]])
        .collect::<Vec<_>>();
    let x = tensor3(position_values, [1, particle_count, 2], &device);
    let s = tensor3(states, [1, particle_count, npa_config.state_dims], &device).require_grad();
    let mut config = test_direct_config(particle_count);
    config.stopgrad_pos = true;
    config.stopgrad_state = false;
    config.grid_eps = grid.eps;
    config.perception_backend = PerceptionRolloutBackend::TiledAdjoint;
    let cube_config = perception_cube_adjoint_config(grid.eps, false, true);
    let feature_grad_inner = Tensor::<InnerBackend, 3>::from_data(
        TensorData::new(feature_weights.clone(), [1, particle_count, feature_dims]),
        &x.clone().inner().device(),
    );
    let prepared_forward = InnerBackend::perception_cube_forward_prepared(
        x.clone().inner(),
        s.clone().inner(),
        cube_config,
    )
    .expect("prepared perception backend")
    .expect("prepared perception forward");
    <InnerBackend as Backend>::sync(&device).expect("prepared perception forward device execution");
    prepared_forward
        .features
        .clone()
        .try_into_data()
        .expect("prepared perception forward feature readback");
    let block_info = prepared_forward
        .block_info
        .clone()
        .try_into_data()
        .expect("prepared perception block-info readback")
        .to_vec::<u32>()
        .expect("prepared perception block-info u32 conversion");
    let cell_count = cube_config.grid_width as usize * cube_config.grid_height as usize;
    let mut assigned_queries = 0usize;
    for block in block_info.chunks_exact(3) {
        let [cell, query_start, query_count] =
            [block[0] as usize, block[1] as usize, block[2] as usize];
        assert!(query_count <= 16, "query block exceeds 16 particles");
        if query_count > 0 {
            assert!(cell < cell_count, "query block cell is out of range");
            assert!(
                query_start + query_count <= particle_count,
                "query block particle range is out of bounds"
            );
        }
        assigned_queries += query_count;
    }
    assert_eq!(assigned_queries, particle_count);
    let prepared_adjoint = InnerBackend::perception_cube_adjoint_prepared(
        x.clone().inner(),
        s.clone().inner(),
        feature_grad_inner.clone(),
        prepared_forward.density,
        prepared_forward.offsets,
        prepared_forward.permutation,
        prepared_forward.block_info,
        prepared_forward.raw_state_gradient,
        prepared_forward.state_gradient_inverse,
        cube_config,
    )
    .expect("prepared perception adjoint backend")
    .expect("prepared perception adjoint");
    <InnerBackend as Backend>::sync(&device).expect("prepared perception adjoint device execution");
    prepared_adjoint
        .state_grad
        .clone()
        .try_into_data()
        .expect("prepared perception adjoint state-gradient readback");
    let recomputed_adjoint = InnerBackend::perception_cube_adjoint(
        x.clone().inner(),
        s.clone().inner(),
        feature_grad_inner,
        cube_config,
    )
    .expect("recomputed perception adjoint backend")
    .expect("recomputed perception adjoint");
    <InnerBackend as Backend>::sync(&device)
        .expect("recomputed perception adjoint device execution");
    recomputed_adjoint
        .state_grad
        .clone()
        .try_into_data()
        .expect("recomputed perception adjoint state-gradient readback");
    let prepared_state = tensor3_vec(prepared_adjoint.state_grad).unwrap();
    let recomputed_state = tensor3_vec(recomputed_adjoint.state_grad).unwrap();
    let prepared_recomputed_diff = max_abs_difference(&prepared_state, &recomputed_state);
    let prepared_reuse_before = PERCEPTION_CUBE_PREPARED_REUSE_HITS.load(Ordering::Relaxed);
    let features = perception_tiled_adjoint_batch(x, s.clone(), config);
    features
        .clone()
        .inner()
        .try_into_data()
        .expect("custom perception autodiff forward feature readback");
    let weights = tensor3(feature_weights, [1, particle_count, feature_dims], &device);
    let loss = features.clone().mul(weights).sum();
    let feature_values = tensor3_vec(features.inner()).unwrap();
    let mut grads = loss.backward();
    <InnerBackend as Backend>::sync(&device)
        .expect("custom perception autodiff backward device execution");
    let state_grad = tensor3_vec(
        s.grad_remove(&mut grads)
            .unwrap_or_else(|| s.clone().inner().zeros_like()),
    )
    .unwrap();

    let (feature_idx, feature_diff, feature_reference, feature_sparse) =
        max_abs_difference_with_index(&reference.features, &feature_values);
    let feature_diff_by_channel = (0..feature_dims)
        .map(|channel| {
            reference
                .features
                .iter()
                .skip(channel)
                .step_by(feature_dims)
                .zip(feature_values.iter().skip(channel).step_by(feature_dims))
                .map(|(reference, sparse)| (reference - sparse).abs())
                .fold(0.0_f32, f32::max)
        })
        .collect::<Vec<_>>();
    let (_, state_diff, state_reference, state_sparse) =
        max_abs_difference_with_index(&reference_adjoint.state, &state_grad);
    let state_relative = state_diff / state_reference.abs().max(state_sparse.abs()).max(1.0e-4);
    let prepared_reference_diff = max_abs_difference(&prepared_state, &reference_adjoint.state);
    let recomputed_reference_diff = max_abs_difference(&recomputed_state, &reference_adjoint.state);
    assert!(
        PERCEPTION_CUBE_PREPARED_REUSE_HITS.load(Ordering::Relaxed) > prepared_reuse_before,
        "sparse perception backward did not reuse its forward grid/density"
    );
    assert!(
        feature_diff < 1.0e-2,
        "sparse cell-plane perception forward diverged from the 1024-particle reference: idx={feature_idx} channel={} reference={feature_reference} sparse={feature_sparse} max_abs_diff={feature_diff} by_channel={feature_diff_by_channel:?}",
        feature_idx % feature_dims,
    );
    assert!(
        prepared_recomputed_diff < 2.0e-5,
        "retained-state perception VJP diverged from recomputed sparse VJP: max_abs_diff={prepared_recomputed_diff} prepared_reference_diff={prepared_reference_diff} recomputed_reference_diff={recomputed_reference_diff} feature_diff={feature_diff} state_diff={state_diff} state_relative={state_relative} reference={state_reference} sparse={state_sparse}"
    );
    assert!(
        state_diff < 6.0e-3 && state_relative < 5.0e-2,
        "sparse cell-plane perception state VJP diverged from the 1024-particle reference: reference={state_reference} sparse={state_sparse} max_abs_diff={state_diff} rel_diff={state_relative}"
    );
}

#[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
#[test]
fn perception_sparse_blocks_support_24_state_channels() {
    let mut npa_config = NpaConfig::growing_2d();
    npa_config.state_dims = 24;
    let grid = crate::upstream_growing_2d_hashgrid();
    let particle_count = 1024;
    let (positions, mut states) = seed_particles_scaled(
        1,
        particle_count,
        npa_config.state_dims,
        npa_config.spatial_dims,
        97,
        crate::ParticleSeed::UniformCircle,
        0.5,
    );
    for (index, state) in states.iter_mut().enumerate() {
        *state = (index as f32 * 0.013_731).sin() * 0.1;
    }
    let options = perception_reference_options(grid.eps);
    let reference = burn_automata_kernels::perceive_with_options(
        &positions,
        &states,
        1,
        particle_count,
        npa_config.state_dims,
        &grid,
        options,
    )
    .unwrap();
    let feature_dims = reference.feature_dims;
    let feature_weights = (0..particle_count * feature_dims)
        .map(|index| (((index * 23) % 29) as f32 - 14.0) * 1.0e-4)
        .collect::<Vec<_>>();
    let reference_adjoint = burn_automata_kernels::perceive_adjoint_with_options(
        &positions,
        &states,
        1,
        particle_count,
        npa_config.state_dims,
        &grid,
        options,
        &feature_weights,
    )
    .unwrap();

    let device = BurnDevice::default();
    let x = tensor3(
        positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect(),
        [1, particle_count, 2],
        &device,
    )
    .inner();
    let s = tensor3(
        states,
        [1, particle_count, npa_config.state_dims],
        &device,
    )
    .inner();
    let cube_config = perception_cube_adjoint_config(grid.eps, false, true);
    let prepared = InnerBackend::perception_cube_forward_prepared(
        x.clone(),
        s.clone(),
        cube_config,
    )
    .expect("24-channel prepared perception backend")
    .expect("24-channel prepared perception forward");
    let feature_values = tensor3_vec(prepared.features.clone()).unwrap();
    let feature_grad = Tensor::<InnerBackend, 3>::from_data(
        TensorData::new(feature_weights, [1, particle_count, feature_dims]),
        &x.device(),
    );
    let adjoint = InnerBackend::perception_cube_adjoint_prepared(
        x,
        s,
        feature_grad,
        prepared.density,
        prepared.offsets,
        prepared.permutation,
        prepared.block_info,
        prepared.raw_state_gradient,
        prepared.state_gradient_inverse,
        cube_config,
    )
    .expect("24-channel prepared perception adjoint backend")
    .expect("24-channel prepared perception adjoint");
    let state_gradient = tensor3_vec(adjoint.state_grad).unwrap();
    let feature_diff = max_abs_difference(&reference.features, &feature_values);
    let (_, state_diff, reference_state, device_state) =
        max_abs_difference_with_index(&reference_adjoint.state, &state_gradient);
    let state_relative = state_diff
        / reference_state
            .abs()
            .max(device_state.abs())
            .max(1.0e-4);
    assert!(
        feature_diff < 1.0e-2,
        "24-channel sparse block forward diverged: max_abs_diff={feature_diff}"
    );
    assert!(
        state_diff < 6.0e-3 && state_relative < 5.0e-2,
        "24-channel sparse block VJP diverged: reference={reference_state} device={device_state} max_abs_diff={state_diff} relative={state_relative}"
    );
}

#[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
#[test]
#[ignore = "opt-in quality-scale perception parity gate"]
fn perception_sparse_blocks_match_planes_at_training_shape() {
    let mut npa_config = NpaConfig::growing_2d();
    npa_config.state_dims = std::env::var("BURN_AUTOMATA_PERCEPTION_PARITY_STATE_DIMS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(npa_config.state_dims);
    let grid = crate::upstream_growing_2d_hashgrid();
    let batch_size = std::env::var("BURN_AUTOMATA_PERCEPTION_PARITY_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8);
    let particle_count = std::env::var("BURN_AUTOMATA_PERCEPTION_PARITY_PARTICLES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4096);
    let (positions, mut states) = seed_particles_scaled(
        batch_size,
        particle_count,
        npa_config.state_dims,
        npa_config.spatial_dims,
        42,
        crate::ParticleSeed::UniformCircle,
        0.2,
    );
    for (index, state) in states.iter_mut().enumerate() {
        *state = (index as f32 * 0.013_731).sin() * 0.1;
    }
    let device = BurnDevice::default();
    let x = tensor3(
        positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect(),
        [batch_size, particle_count, 2],
        &device,
    )
    .inner();
    let s = tensor3(
        states,
        [batch_size, particle_count, npa_config.state_dims],
        &device,
    )
    .inner();
    let feature_dims = npa_config.perception_dims();
    let feature_grad = tensor3(
        (0..batch_size * particle_count * feature_dims)
            .map(|idx| (((idx * 17) % 19) as f32 - 9.0) * 1.0e-4)
            .collect(),
        [batch_size, particle_count, feature_dims],
        &device,
    )
    .inner();
    let cube_config = perception_cube_adjoint_config(grid.eps, false, true);

    let prepared =
        InnerBackend::perception_cube_forward_prepared(x.clone(), s.clone(), cube_config)
            .expect("prepared perception backend")
            .expect("prepared perception forward");
    let plane = InnerBackend::perception_cube_forward(x.clone(), s.clone(), cube_config)
        .expect("plane perception backend")
        .expect("plane perception forward");
    <InnerBackend as Backend>::sync(&device).expect("perception forward execution");

    let block_info = prepared
        .block_info
        .clone()
        .try_into_data()
        .expect("prepared perception block-info readback")
        .to_vec::<u32>()
        .expect("prepared perception block-info u32 conversion");
    let permutation = prepared
        .permutation
        .clone()
        .try_into_data()
        .expect("prepared perception permutation readback")
        .to_vec::<u32>()
        .expect("prepared perception permutation u32 conversion");
    let cell_count = cube_config.grid_width as usize * cube_config.grid_height as usize;
    let mut assigned_queries = vec![0usize; batch_size];
    let mut active_blocks = 0usize;
    for block in block_info.chunks_exact(3) {
        let query_count = block[2] as usize;
        if query_count == 0 {
            continue;
        }
        let batch = block[0] as usize / cell_count;
        assert!(batch < batch_size, "query block batch is out of range");
        assigned_queries[batch] += query_count;
        active_blocks += 1;
    }
    for (batch, assigned_queries) in assigned_queries.into_iter().enumerate() {
        assert_eq!(
            assigned_queries, particle_count,
            "query blocks do not cover batch {batch}"
        );
    }

    let prepared_features = tensor3_vec(prepared.features.clone()).unwrap();
    let plane_features = tensor3_vec(plane.features).unwrap();
    let (feature_idx, feature_diff, plane_feature, prepared_feature) =
        max_abs_difference_with_index(&plane_features, &prepared_features);
    let feature_relative =
        feature_diff / plane_feature.abs().max(prepared_feature.abs()).max(1.0e-4);

    let prepared_adjoint = InnerBackend::perception_cube_adjoint_prepared(
        x.clone(),
        s.clone(),
        feature_grad.clone(),
        prepared.density,
        prepared.offsets,
        prepared.permutation,
        prepared.block_info,
        prepared.raw_state_gradient,
        prepared.state_gradient_inverse,
        cube_config,
    )
    .expect("prepared perception adjoint backend")
    .expect("prepared perception adjoint");
    let plane_adjoint = InnerBackend::perception_cube_adjoint(x, s, feature_grad, cube_config)
        .expect("plane perception adjoint backend")
        .expect("plane perception adjoint");
    <InnerBackend as Backend>::sync(&device).expect("perception adjoint execution");
    let prepared_state = tensor3_vec(prepared_adjoint.state_grad).unwrap();
    let plane_state = tensor3_vec(plane_adjoint.state_grad).unwrap();
    let (state_idx, state_diff, plane_state_value, prepared_state_value) =
        max_abs_difference_with_index(&plane_state, &prepared_state);
    let state_relative = state_diff
        / plane_state_value
            .abs()
            .max(prepared_state_value.abs())
            .max(1.0e-4);
    let state_dims = npa_config.state_dims;
    let mut mismatch_by_batch = vec![0usize; batch_size];
    let mut missing_by_batch = vec![0usize; batch_size];
    let mut mismatch_by_channel = vec![0usize; state_dims];
    let mut missing_by_particle = vec![0usize; batch_size * particle_count];
    for (idx, (&plane_value, &prepared_value)) in
        plane_state.iter().zip(&prepared_state).enumerate()
    {
        if (plane_value - prepared_value).abs() > 2.0e-3 {
            let batch = idx / (particle_count * state_dims);
            let channel = idx % state_dims;
            mismatch_by_batch[batch] += 1;
            mismatch_by_channel[channel] += 1;
            if prepared_value == 0.0 && plane_value != 0.0 {
                missing_by_batch[batch] += 1;
                missing_by_particle[idx / state_dims] += 1;
            }
        }
    }
    let mut affected_blocks = Vec::new();
    for (block, info) in block_info.chunks_exact(3).enumerate() {
        let query_start = info[1] as usize;
        let query_count = info[2] as usize;
        if query_count == 0 {
            continue;
        }
        let packed_cell = info[0] as usize;
        let batch = packed_cell / cell_count;
        let cell = packed_cell - batch * cell_count;
        let missing = (0..query_count)
            .map(|query| {
                let particle = permutation[batch * particle_count + query_start + query] as usize;
                missing_by_particle[batch * particle_count + particle]
            })
            .sum::<usize>();
        if missing > 0 {
            affected_blocks.push((batch, block, cell, query_count, missing));
        }
    }

    println!(
        "quality-scale perception parity batch={batch_size} particles={particle_count} active_blocks={active_blocks} launched_blocks={} active_ratio={:.3} feature_idx={feature_idx} feature_diff={feature_diff:.9e} feature_relative={feature_relative:.9e} plane_feature={plane_feature:.9e} prepared_feature={prepared_feature:.9e} state_idx={state_idx} state_diff={state_diff:.9e} state_relative={state_relative:.9e} plane_state={plane_state_value:.9e} prepared_state={prepared_state_value:.9e} mismatch_by_batch={mismatch_by_batch:?} missing_by_batch={missing_by_batch:?} mismatch_by_channel={mismatch_by_channel:?} affected_blocks={} first_affected_blocks={:?}",
        block_info.len() / 3,
        active_blocks as f64 / (block_info.len() / 3) as f64,
        affected_blocks.len(),
        &affected_blocks[..affected_blocks.len().min(24)],
    );
    let feature_tolerance =
        2.0e-3 + 5.0e-3 * plane_feature.abs().max(prepared_feature.abs());
    assert!(
        feature_diff < feature_tolerance,
        "quality-scale block forward diverged from sparse-plane forward"
    );
    let state_tolerance =
        2.0e-3 + 5.0e-3 * plane_state_value.abs().max(prepared_state_value.abs());
    assert!(
        state_diff < state_tolerance,
        "quality-scale block VJP diverged from sparse-plane VJP"
    );
}

#[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
#[test]
#[ignore = "opt-in GPU perception throughput benchmark"]
fn benchmark_perception_sparse_grid_forward_and_state_vjp() {
    let mut npa_config = NpaConfig::growing_2d();
    npa_config.state_dims = std::env::var("BURN_AUTOMATA_PERCEPTION_BENCH_STATE_DIMS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(npa_config.state_dims);
    let grid = crate::upstream_growing_2d_hashgrid();
    let device = BurnDevice::default();
    let inner_device = device.clone();
    let batch_size = std::env::var("BURN_AUTOMATA_PERCEPTION_BENCH_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let seed_scale = std::env::var("BURN_AUTOMATA_PERCEPTION_BENCH_SEED_SCALE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.5);
    let repeats = std::env::var("BURN_AUTOMATA_PERCEPTION_BENCH_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(50);
    let particles = std::env::var("BURN_AUTOMATA_PERCEPTION_BENCH_PARTICLES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map_or_else(|| vec![512usize, 1024, 2048, 4096], |value| vec![value]);
    let prepared_only =
        std::env::var_os("BURN_AUTOMATA_PERCEPTION_BENCH_PREPARED_ONLY").is_some();
    for particle_count in particles {
        let (positions, states) = seed_particles_scaled(
            batch_size,
            particle_count,
            npa_config.state_dims,
            npa_config.spatial_dims,
            7331 + particle_count as u64,
            crate::ParticleSeed::UniformCircle,
            seed_scale,
        );
        let x = tensor3(
            positions
                .iter()
                .flat_map(|position| [position[0], position[1]])
                .collect(),
            [batch_size, particle_count, 2],
            &device,
        )
        .inner();
        let s = tensor3(
            states,
            [batch_size, particle_count, npa_config.state_dims],
            &device,
        )
        .inner();
        let feature_dims = npa_config.perception_dims();
        let feature_grad = tensor3(
            (0..batch_size * particle_count * feature_dims)
                .map(|idx| (((idx * 17) % 19) as f32 - 9.0) * 1.0e-4)
                .collect(),
            [batch_size, particle_count, feature_dims],
            &device,
        )
        .inner();

        let modes = [
            ("all_pairs", u32::MAX, false),
            ("sparse_grid", 1u32, false),
            ("sparse_prepared", 1u32, true),
        ];
        for (mode, sparse_grid_min_particles, reuse_forward) in modes {
            if prepared_only && !reuse_forward {
                continue;
            }
            let mut cube_config = perception_cube_adjoint_config(grid.eps, false, true);
            cube_config.sparse_grid_min_particles = sparse_grid_min_particles;
            let run = || {
                if reuse_forward {
                    let forward = InnerBackend::perception_cube_forward_prepared(
                        x.clone(),
                        s.clone(),
                        cube_config,
                    )
                    .expect("CubeCL prepared perception forward backend")
                    .expect("CubeCL prepared perception forward");
                    let adjoint = InnerBackend::perception_cube_adjoint_prepared(
                        x.clone(),
                        s.clone(),
                        feature_grad.clone(),
                        forward.density.clone(),
                        forward.offsets.clone(),
                        forward.permutation.clone(),
                        forward.block_info.clone(),
                        forward.raw_state_gradient.clone(),
                        forward.state_gradient_inverse.clone(),
                        cube_config,
                    )
                    .expect("CubeCL prepared perception adjoint backend")
                    .expect("CubeCL prepared perception adjoint");
                    (forward.features, adjoint)
                } else {
                    let forward =
                        InnerBackend::perception_cube_forward(x.clone(), s.clone(), cube_config)
                            .expect("CubeCL perception forward backend")
                            .expect("CubeCL perception forward");
                    let adjoint = InnerBackend::perception_cube_adjoint(
                        x.clone(),
                        s.clone(),
                        feature_grad.clone(),
                        cube_config,
                    )
                    .expect("CubeCL perception adjoint backend")
                    .expect("CubeCL perception adjoint");
                    (forward.features, adjoint)
                }
            };
            let warmup = run();
            <InnerBackend as Backend>::sync(&inner_device).unwrap();
            drop(warmup);

            let start = Instant::now();
            for _ in 0..repeats {
                let output = run();
                <InnerBackend as Backend>::sync(&inner_device).unwrap();
                drop(output);
            }
            let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0;
            println!(
                "perception_cube_bench mode={mode} batch={batch_size} particles={particle_count} seed_scale={seed_scale} repeats={repeats} mean_forward_vjp_ms={:.6} particle_steps_per_sec={:.0}",
                elapsed_ms / repeats as f64,
                (batch_size * particle_count) as f64 * repeats as f64
                    / (elapsed_ms / 1_000.0),
            );
        }
    }
}

#[test]
fn burn_target_splat_loss_matches_reference_cpu_fixture() {
    let npa_config = NpaConfig::growing_2d();
    let grid = burn_automata_kernels::HashGridConfig::growing_2d();
    let target = crate::TargetImage2d {
        source_width: 16,
        source_height: 16,
        positions: vec![[-0.35, 0.25], [0.2, 0.05], [0.45, -0.3]],
        colors: vec![[0.1, 0.8, 0.2], [0.7, 0.3, 0.1], [0.2, 0.4, 0.9]],
        pixel_size: 2.0 / 16.0,
        threshold: 0.05,
        aabb: [-1.0, 1.0, -1.0, 1.0],
    };
    let loss_config = crate::Target2dLossConfig {
        image_size: 16,
        sigma: 1.0,
        center: true,
        foreground_density_loss_weight: 0.5,
        composited_rgb_loss_weight: 0.75,
        render_rgb_loss_weight: 1.25,
        displacement_regularizer_weight: 0.0,
        overflow_regularizer_weight: 0.0,
        bound_regularizer_weight: 0.0,
        ..crate::Target2dLossConfig::default()
    };
    let (mut positions, mut states) = seed_particles_scaled(
        1,
        4,
        npa_config.state_dims,
        npa_config.spatial_dims,
        29,
        crate::ParticleSeed::UniformCircle,
        0.3,
    );
    // Keep one partially visible off-image Gaussian in the parity fixture. This
    // exercises upstream's normalize-full-patch-before-clipping semantics.
    positions[0][0] = 2.0;
    for particle in 0..4 {
        let base = particle * npa_config.state_dims + npa_config.state_dims - 3;
        states[base] = -0.3 + particle as f32 * 0.1;
        states[base + 1] = 0.2 - particle as f32 * 0.05;
        states[base + 2] = -0.1 + particle as f32 * 0.07;
    }
    let reference_output = crate::target_2d_loss_with_adjoint(
        &positions,
        &states,
        1,
        4,
        npa_config.state_dims,
        &target,
        loss_config,
        0.0,
        0,
    )
    .unwrap();
    let reference = reference_output.report;

    let device = BurnDevice::default();
    let pixels = loss_config.image_size * loss_config.image_size;
    let render = render_target_2d_splat(&target, loss_config).unwrap();
    let foreground = target_2d_foreground_mask(&target, loss_config).unwrap();
    let foreground_scale = pixels as f32 / foreground.iter().sum::<f32>().max(1.0);
    let target_mean = target.mean_position();
    let target = BurnTargetExample {
        target_rgb: tensor(render.rgb, [pixels, 3], &device),
        target_density: tensor(render.density, [pixels, 1], &device),
        target_foreground: tensor(foreground, [pixels, 1], &device),
        target_foreground_scale: foreground_scale,
        target_mean: tensor([target_mean[0], target_mean[1]].to_vec(), [1, 2], &device),
        target_positions: tensor(
            target
                .positions
                .iter()
                .flat_map(|position| [position[0], position[1]])
                .collect(),
            [target.positions.len(), 2],
            &device,
        ),
        pixel_xy: tensor(
            pixel_xy_values(loss_config.image_size),
            [pixels, 2],
            &device,
        ),
        pixel_size: 2.0 / 16.0,
        target_points: 3,
        particle_count: 4,
        update_prob: 1.0,
        seed_scale: 0.3,
        target_cpu: target.clone(),
    };
    let model = NpaModel::upstream_seeded(npa_config.clone(), 29);
    let adapter = BurnAdapterParams::from_adapter(
        &NpaLowRankAdapter::zeros(&npa_config, 1, 1.0),
        &model,
        &device,
    )
    .unwrap();
    let config = DirectBasisTrainConfig {
        steps: 0,
        report_interval: 1,
        example_batch_size: 1,
        tbptt_chunk_steps: 1,
        loss_on_final_chunk_only: false,
        use_particle_pool: false,
        pool_size: 0,
        inject_seed_interval: 0,
        brush_size: 0.0,
        stopgrad_pos: npa_config.stopgrad_pos,
        stopgrad_state: npa_config.stopgrad_state,
        rollout_particles: 4,
        rollout_step_min: 1,
        rollout_steps: 1,
        update_prob: 1.0,
        seed: 29,
        seed_scale: 0.3,
        seed_mode: crate::ParticleSeed::UniformCircle,
        grid_eps: grid.eps,
        motion_scale: npa_config.alpha * npa_config.motion_eps(grid.eps),
        loss_config,
        target2d_loss_backend: Target2dLossBackend::Dense,
        perception_backend: PerceptionRolloutBackend::Dense,
        per_parameter_grad_normalization: false,
        base_sgd: SgdConfig {
            learning_rate: 0.0,
            weight_decay: 0.0,
            grad_clip_norm: 0.0,
        },
        adapter_sgd: SgdConfig {
            learning_rate: 0.0,
            weight_decay: 0.0,
            grad_clip_norm: 0.0,
        },
        adapter_l2_weight: 0.0,
        update_base: true,
        eval_examples: 1,
        eval_interval: 0,
        eval_batch_size: 1,
        eval_seed: 29,
        system_memory_budget_gb: None,
        gpu_memory_budget_gb: None,
        max_dense_train_particles: 4,
        max_dense_chunk_floats: 1_000_000,
        max_splat_chunk_floats: 1_000_000,
    };
    let x = tracked_tensor(
        positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect(),
        [4, 2],
        &device,
    );
    let s = tracked_tensor(states.clone(), [4, npa_config.state_dims], &device);
    let loss = target_splat_loss(
        &x,
        &s,
        &target,
        config,
        &adapter,
        Tensor::<BurnBackend, 1>::zeros([1], &device),
    );

    let burn_total = loss.total.clone().inner().into_scalar();
    let burn_splat = loss.splat.clone().inner().into_scalar();
    let burn_color = loss.color.clone().inner().into_scalar();
    let burn_density = loss.density.clone().inner().into_scalar();

    assert!(
        (burn_total - reference.total_loss).abs() < 1.0e-4,
        "Burn total target2d loss diverged from CPU reference: burn={burn_total} reference={}",
        reference.total_loss
    );
    assert!(
        (burn_splat - reference.splat_loss).abs() < 1.0e-4,
        "Burn splat target2d loss diverged from CPU reference: burn={burn_splat} reference={}",
        reference.splat_loss
    );
    assert!(
        (burn_color - reference.color_loss).abs() < 1.0e-4,
        "Burn color target2d loss diverged from CPU reference: burn={burn_color} reference={}",
        reference.color_loss
    );
    assert!(
        (burn_density - reference.density_loss).abs() < 1.0e-4,
        "Burn density target2d loss diverged from CPU reference: burn={burn_density} reference={}",
        reference.density_loss
    );

    let mut grads = loss.total.backward();
    let burn_x_grad = tensor_vec(
        x.grad_remove(&mut grads)
            .unwrap_or_else(|| x.clone().inner().zeros_like()),
    )
    .unwrap();
    let burn_s_grad = tensor_vec(
        s.grad_remove(&mut grads)
            .unwrap_or_else(|| s.clone().inner().zeros_like()),
    )
    .unwrap();
    let max_position_grad_diff = burn_x_grad
        .chunks_exact(2)
        .zip(&reference_output.position_gradients)
        .flat_map(|(burn, reference)| {
            [
                (burn[0] - reference[0]).abs(),
                (burn[1] - reference[1]).abs(),
            ]
        })
        .fold(0.0_f32, f32::max);
    let max_state_grad_diff = burn_s_grad
        .iter()
        .zip(&reference_output.state_gradients)
        .map(|(burn, reference)| (burn - reference).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_position_grad_diff < 2.0e-3,
        "Burn target2d position gradient diverged from CPU reference: max_abs_diff={max_position_grad_diff}"
    );
    assert!(
        max_state_grad_diff < 2.0e-3,
        "Burn target2d state gradient diverged from CPU reference: max_abs_diff={max_state_grad_diff}"
    );

    let x3 = tensor3(
        positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect(),
        [1, 4, 2],
        &device,
    )
    .require_grad();
    let s3 = tensor3(states.clone(), [1, 4, npa_config.state_dims], &device).require_grad();
    let adapter_batch = BurnAdapterBatch::from_indices(std::slice::from_ref(&adapter), &[0]);
    let tiled_config = DirectBasisTrainConfig {
        target2d_loss_backend: Target2dLossBackend::TiledAdjoint,
        perception_backend: PerceptionRolloutBackend::Dense,
        ..config
    };
    let tiled_loss = target_splat_loss_batch_vector_selected(
        &x3,
        &s3,
        std::slice::from_ref(&target),
        &[0],
        tiled_config,
        &adapter_batch,
        Tensor::<BurnBackend, 1>::zeros([1], &device),
    )
    .unwrap();
    let tiled_scalars = loss_vector_scalars(tiled_loss.clone()).unwrap();
    let tiled_scalar = tiled_scalars[0];
    assert!(
        (tiled_scalar.total - reference.total_loss).abs() < 1.0e-4,
        "tiled-adjoint total target2d loss diverged from CPU reference: tiled={} reference={}",
        tiled_scalar.total,
        reference.total_loss
    );
    assert!(
        (tiled_scalar.splat - reference.splat_loss).abs() < 1.0e-4,
        "tiled-adjoint splat target2d loss diverged from CPU reference: tiled={} reference={}",
        tiled_scalar.splat,
        reference.splat_loss
    );
    assert!(
        (tiled_scalar.color - reference.color_loss).abs() < 1.0e-4,
        "tiled-adjoint color target2d loss diverged from CPU reference: tiled={} reference={}",
        tiled_scalar.color,
        reference.color_loss
    );
    assert!(
        (tiled_scalar.density - reference.density_loss).abs() < 1.0e-4,
        "tiled-adjoint density target2d loss diverged from CPU reference: tiled={} reference={}",
        tiled_scalar.density,
        reference.density_loss
    );

    let mut grads = tiled_loss.total.sum().backward();
    let tiled_x_grad = tensor3_vec(
        x3.grad_remove(&mut grads)
            .unwrap_or_else(|| x3.clone().inner().zeros_like()),
    )
    .unwrap();
    let tiled_s_grad = tensor3_vec(
        s3.grad_remove(&mut grads)
            .unwrap_or_else(|| s3.clone().inner().zeros_like()),
    )
    .unwrap();
    let max_position_grad_diff = tiled_x_grad
        .chunks_exact(2)
        .zip(&reference_output.position_gradients)
        .flat_map(|(burn, reference)| {
            [
                (burn[0] - reference[0]).abs(),
                (burn[1] - reference[1]).abs(),
            ]
        })
        .fold(0.0_f32, f32::max);
    let max_state_grad_diff = tiled_s_grad
        .iter()
        .zip(&reference_output.state_gradients)
        .map(|(burn, reference)| (burn - reference).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_position_grad_diff < 1.0e-6,
        "tiled-adjoint target2d position gradient diverged from CPU reference: max_abs_diff={max_position_grad_diff}"
    );
    assert!(
        max_state_grad_diff < 1.0e-6,
        "tiled-adjoint target2d state gradient diverged from CPU reference: max_abs_diff={max_state_grad_diff}"
    );

    let base_x3 = tensor3(
        positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect(),
        [1, 4, 2],
        &device,
    )
    .require_grad();
    let base_s3 =
        tensor3(states.clone(), [1, 4, npa_config.state_dims], &device).require_grad();
    let base_only_loss = target_splat_loss_batch_vector_base_only_selected(
        &base_x3,
        &base_s3,
        std::slice::from_ref(&target),
        &[0],
        tiled_config,
        Tensor::<BurnBackend, 1>::zeros([1], &device),
    )
    .unwrap();
    let base_only_scalar = loss_vector_scalars(base_only_loss.clone()).unwrap()[0];
    assert!(
        (base_only_scalar.total - tiled_scalar.total).abs() < 1.0e-6,
        "base-only tiled target2d total diverged from adapter-batch path: base={} adapter={}",
        base_only_scalar.total,
        tiled_scalar.total,
    );
    let mut base_grads = base_only_loss.total.sum().backward();
    let base_x_grad = tensor3_vec(
        base_x3
            .grad_remove(&mut base_grads)
            .unwrap_or_else(|| base_x3.clone().inner().zeros_like()),
    )
    .unwrap();
    let base_s_grad = tensor3_vec(
        base_s3
            .grad_remove(&mut base_grads)
            .unwrap_or_else(|| base_s3.clone().inner().zeros_like()),
    )
    .unwrap();
    assert!(max_abs_difference(&base_x_grad, &tiled_x_grad) < 1.0e-6);
    assert!(max_abs_difference(&base_s_grad, &tiled_s_grad) < 1.0e-6);

    let fixed_dense_x3 = tensor3(
        positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect(),
        [1, 4, 2],
        &device,
    )
    .require_grad();
    let fixed_dense_s3 =
        tensor3(states.clone(), [1, 4, npa_config.state_dims], &device).require_grad();
    let fixed_dense_loss = target_splat_loss_batch_vector_base_only_selected(
        &fixed_dense_x3,
        &fixed_dense_s3,
        std::slice::from_ref(&target),
        &[0],
        config,
        Tensor::<BurnBackend, 1>::zeros([1], &device),
    )
    .unwrap();
    let fixed_dense_scalar = loss_vector_scalars(fixed_dense_loss.clone()).unwrap()[0];
    let mut fixed_dense_grads = fixed_dense_loss.total.sum().backward();
    let fixed_dense_x_grad = tensor3_vec(
        fixed_dense_x3
            .grad_remove(&mut fixed_dense_grads)
            .unwrap_or_else(|| fixed_dense_x3.clone().inner().zeros_like()),
    )
    .unwrap();
    let fixed_dense_s_grad = tensor3_vec(
        fixed_dense_s3
            .grad_remove(&mut fixed_dense_grads)
            .unwrap_or_else(|| fixed_dense_s3.clone().inner().zeros_like()),
    )
    .unwrap();

    let adaptive_x3 = tensor3(
        positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect(),
        [1, 4, 2],
        &device,
    )
    .require_grad();
    let adaptive_s3 =
        tensor3(states.clone(), [1, 4, npa_config.state_dims], &device).require_grad();
    let adaptive_loss = adaptive_target_splat_loss_batch_vector_base_only_selected(
        &adaptive_x3,
        &adaptive_s3,
        std::slice::from_ref(&target),
        &[0],
        config,
        tensor(vec![1.0; 4], [1, 4], &device),
        tensor(vec![target.pixel_size; 4], [1, 4], &device),
        tensor(
            vec![target.target_points as f32 / 4.0; 4],
            [1, 4],
            &device,
        ),
        Tensor::<BurnBackend, 1>::zeros([1], &device),
    )
    .unwrap();
    let adaptive_scalar = loss_vector_scalars(adaptive_loss.clone()).unwrap()[0];
    assert!(
        (adaptive_scalar.total - fixed_dense_scalar.total).abs() < 1.0e-6,
        "equal-scale adaptive Target2D loss diverged from fixed loss: adaptive={} fixed={}",
        adaptive_scalar.total,
        fixed_dense_scalar.total,
    );
    let mut adaptive_grads = adaptive_loss.total.sum().backward();
    let adaptive_x_grad = tensor3_vec(
        adaptive_x3
            .grad_remove(&mut adaptive_grads)
            .unwrap_or_else(|| adaptive_x3.clone().inner().zeros_like()),
    )
    .unwrap();
    let adaptive_s_grad = tensor3_vec(
        adaptive_s3
            .grad_remove(&mut adaptive_grads)
            .unwrap_or_else(|| adaptive_s3.clone().inner().zeros_like()),
    )
    .unwrap();
    assert!(
        max_abs_difference(&adaptive_x_grad, &fixed_dense_x_grad) < 1.0e-6,
        "equal-scale adaptive Target2D position VJP diverged from fixed loss"
    );
    assert!(
        max_abs_difference(&adaptive_s_grad, &fixed_dense_s_grad) < 1.0e-6,
        "equal-scale adaptive Target2D state VJP diverged from fixed loss"
    );

    #[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
    {
        let fine_units = [1.0_f32, 4.0, 2.0, 1.0];
        let reference_particles = fine_units.iter().sum::<f32>();
        let represented_measure = fine_units.to_vec();
        let pixel_size = fine_units
            .iter()
            .map(|units| target.pixel_size * units.sqrt())
            .collect::<Vec<_>>();
        let output_scale =
            vec![target.target_points as f32 / reference_particles; fine_units.len()];
        let evaluate = |loss_backend, require_grad: bool| {
            let mut local_config = config;
            local_config.target2d_loss_backend = loss_backend;
            let x = tensor3(
                positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]])
                    .collect(),
                [1, 4, 2],
                &device,
            );
            let s = tensor3(states.clone(), [1, 4, npa_config.state_dims], &device);
            let x = if require_grad { x.require_grad() } else { x };
            let s = if require_grad { s.require_grad() } else { s };
            let loss = adaptive_target_splat_loss_batch_vector_base_only_selected(
                &x,
                &s,
                std::slice::from_ref(&target),
                &[0],
                local_config,
                tensor(represented_measure.clone(), [1, 4], &device),
                tensor(pixel_size.clone(), [1, 4], &device),
                tensor(output_scale.clone(), [1, 4], &device),
                Tensor::<BurnBackend, 1>::zeros([1], &device),
            )
            .unwrap();
            let scalar = loss_vector_scalars(loss.clone()).unwrap()[0];
            let mut grads = loss.total.sum().backward();
            let x_grad = tensor3_vec(
                x.grad_remove(&mut grads)
                    .unwrap_or_else(|| x.clone().inner().zeros_like()),
            )
            .unwrap();
            let s_grad = tensor3_vec(
                s.grad_remove(&mut grads)
                    .unwrap_or_else(|| s.clone().inner().zeros_like()),
            )
            .unwrap();
            (scalar, x_grad, s_grad)
        };
        let dense = evaluate(Target2dLossBackend::Dense, true);
        let tiled = evaluate(Target2dLossBackend::TiledAdjoint, true);
        assert!(
            (dense.0.total - tiled.0.total).abs() < 2.0e-4,
            "variable-footprint Target2D mismatch: dense total/splat/color/density={}/{}/{}/{} tiled={}/{}/{}/{}",
            dense.0.total,
            dense.0.splat,
            dense.0.color,
            dense.0.density,
            tiled.0.total,
            tiled.0.splat,
            tiled.0.color,
            tiled.0.density,
        );
        assert!(
            max_abs_difference(&dense.1, &tiled.1) < 2.0e-3,
            "variable-footprint Target2D position VJP mismatch: max_abs_diff={}",
            max_abs_difference(&dense.1, &tiled.1),
        );
        assert!(
            max_abs_difference(&dense.2, &tiled.2) < 2.0e-3,
            "variable-footprint Target2D state VJP mismatch: max_abs_diff={}",
            max_abs_difference(&dense.2, &tiled.2),
        );
    }
}

#[test]
fn zero_primal_explicit_adjoint_preserves_primal_and_nonlinear_vjp() {
    let device = BurnDevice::default();
    let input = tensor3(vec![100_000_000.0], [1, 1, 1], &device).require_grad();
    let gradient = tensor3(vec![1.0], [1, 1, 1], &device);
    let accurate = tensor1(vec![0.25], [1], &device);
    let loss = accurate.detach() + zero_primal_linear_adjoint_3d(&input, gradient);

    let primal = tensor1_vec(loss.clone().inner()).unwrap()[0];
    assert!(
        (primal - 0.25).abs() < 1.0e-7,
        "accurate explicit-adjoint primal was not preserved: {primal}"
    );

    let mut gradients = loss.log1p().sum().backward();
    let gradient = tensor3_vec(
        input
            .grad_remove(&mut gradients)
            .unwrap_or_else(|| input.clone().inner().zeros_like()),
    )
    .unwrap()[0];
    assert!(
        (gradient - 0.8).abs() < 1.0e-6,
        "nonlinear VJP used the cancellation-prone surrogate primal: {gradient}"
    );
}

#[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
#[test]
fn target2d_cube_batched_tail_matches_dense_primal_and_vjp() {
    const BATCHES: usize = 8;
    const PARTICLES: usize = 64;

    let device = BurnDevice::default();
    let target = test_burn_target(&device, 1.0, 0.2);
    let indices = vec![0; BATCHES];
    let state_dims = NpaConfig::growing_2d().state_dims;
    let mut positions = Vec::with_capacity(BATCHES * PARTICLES * 2);
    let mut states = vec![0.0; BATCHES * PARTICLES * state_dims];
    for batch in 0..BATCHES {
        for particle in 0..PARTICLES {
            let angle = particle as f32 * std::f32::consts::TAU / PARTICLES as f32;
            let radius = 0.05 + 0.01 * (particle % 5) as f32;
            positions.extend([
                radius * angle.cos() + batch as f32 * 0.002,
                radius * angle.sin() - batch as f32 * 0.001,
            ]);
            let color = (batch * PARTICLES + particle) * state_dims + state_dims - 3;
            states[color] = -0.2 + batch as f32 * 0.01;
            states[color + 1] = 0.1 + particle as f32 * 0.001;
            states[color + 2] = -0.05;
        }
    }

    let represented_measure = vec![1.0; BATCHES * PARTICLES];
    let particle_pixel_size = vec![0.2; BATCHES * PARTICLES];
    let particle_output_scale = vec![1.0 / PARTICLES as f32; BATCHES * PARTICLES];
    let mut base_config = test_direct_config(PARTICLES);
    base_config.example_batch_size = BATCHES;
    base_config.loss_config.image_size = 1;
    base_config.loss_config.center = false;
    base_config.loss_config.splat_loss_weight = 2.0;
    base_config.loss_config.color_loss_weight = 5.0;
    base_config.loss_config.density_loss_weight = 1.0;
    base_config.loss_config.background_density_loss_weight = 0.0;
    base_config.loss_config.foreground_density_loss_weight = 0.5;
    base_config.loss_config.composited_rgb_loss_weight = 0.75;
    base_config.loss_config.render_rgb_loss_weight = 1.25;
    base_config.loss_config.shape_chamfer_loss_weight = 0.0;
    base_config.loss_config.displacement_regularizer_weight = 0.0;
    base_config.loss_config.overflow_regularizer_weight = 0.0;
    base_config.loss_config.bound_regularizer_weight = 0.0;

    let evaluate = |backend| {
        let x = tensor3(
            positions.clone(),
            [BATCHES, PARTICLES, 2],
            &device,
        )
        .require_grad();
        let s = tensor3(
            states.clone(),
            [BATCHES, PARTICLES, state_dims],
            &device,
        )
        .require_grad();
        let loss = adaptive_target_splat_loss_batch_vector_base_only_selected(
            &x,
            &s,
            std::slice::from_ref(&target),
            &indices,
            DirectBasisTrainConfig {
                target2d_loss_backend: backend,
                ..base_config
            },
            tensor(
                represented_measure.clone(),
                [BATCHES, PARTICLES],
                &device,
            ),
            tensor(
                particle_pixel_size.clone(),
                [BATCHES, PARTICLES],
                &device,
            ),
            tensor(
                particle_output_scale.clone(),
                [BATCHES, PARTICLES],
                &device,
            ),
            Tensor::<BurnBackend, 1>::zeros([BATCHES], &device),
        )
        .unwrap();
        let splat = tensor1_vec(loss.splat.clone().inner()).unwrap();
        let total = tensor1_vec(loss.total.clone().inner()).unwrap();
        let mut gradients = loss.total.sum().backward();
        let x_grad = tensor3_vec(
            x.grad_remove(&mut gradients)
                .unwrap_or_else(|| x.clone().inner().zeros_like()),
        )
        .unwrap();
        let s_grad = tensor3_vec(
            s.grad_remove(&mut gradients)
                .unwrap_or_else(|| s.clone().inner().zeros_like()),
        )
        .unwrap();
        (splat, total, x_grad, s_grad)
    };

    let dense = evaluate(Target2dLossBackend::Dense);
    let tiled = evaluate(Target2dLossBackend::TiledAdjoint);
    for row in 0..BATCHES {
        assert!(tiled.0[row].is_finite(), "non-finite tiled splat row {row}");
        assert!(
            (dense.0[row] - tiled.0[row]).abs() < 2.0e-4,
            "batched Target2D splat row {row} diverged: dense={} tiled={}",
            dense.0[row],
            tiled.0[row],
        );
        assert!(
            (dense.1[row] - tiled.1[row]).abs() < 2.0e-4,
            "batched Target2D primal row {row} diverged: dense={} tiled={}",
            dense.1[row],
            tiled.1[row],
        );
    }
    assert!(
        max_abs_difference(&dense.2, &tiled.2) < 2.0e-3,
        "batched Target2D position VJP diverged"
    );
    assert!(
        max_abs_difference(&dense.3, &tiled.3) < 2.0e-3,
        "batched Target2D state VJP diverged"
    );
}

#[test]
fn adaptive_render_only_objective_matches_reported_rgb_mse() {
    const BATCHES: usize = 4;
    const PARTICLES: usize = 3_070;
    const IMAGE_SIZE: usize = 256;

    let device = BurnDevice::default();
    let indices = vec![0; BATCHES];
    let state_dims = NpaConfig::growing_2d().state_dims;
    let target_cpu = crate::TargetImage2d {
        source_width: IMAGE_SIZE,
        source_height: IMAGE_SIZE,
        positions: vec![[-0.35, 0.25], [0.2, 0.05], [0.45, -0.3]],
        colors: vec![[0.1, 0.8, 0.2], [0.7, 0.3, 0.1], [0.2, 0.4, 0.9]],
        pixel_size: 2.0 / IMAGE_SIZE as f32,
        threshold: 0.05,
        aabb: [-1.0, 1.0, -1.0, 1.0],
    };
    let mut config = test_direct_config(PARTICLES);
    config.example_batch_size = BATCHES;
    config.target2d_loss_backend = Target2dLossBackend::Dense;
    config.loss_config.image_size = IMAGE_SIZE;
    config.loss_config.center = false;
    config.loss_config.splat_loss_weight = 1.0;
    config.loss_config.color_loss_weight = 0.0;
    config.loss_config.density_loss_weight = 0.0;
    config.loss_config.background_density_loss_weight = 0.0;
    config.loss_config.foreground_density_loss_weight = 0.0;
    config.loss_config.composited_rgb_loss_weight = 0.0;
    config.loss_config.render_rgb_loss_weight = 1.0;
    config.loss_config.shape_chamfer_loss_weight = 0.0;
    config.loss_config.displacement_regularizer_weight = 0.0;
    config.loss_config.overflow_regularizer_weight = 0.0;
    config.loss_config.bound_regularizer_weight = 0.0;
    config.max_splat_chunk_floats = 512 * 1024;
    let pixels = IMAGE_SIZE * IMAGE_SIZE;
    let render = render_target_2d_splat(&target_cpu, config.loss_config).unwrap();
    let foreground = target_2d_foreground_mask(&target_cpu, config.loss_config).unwrap();
    let foreground_scale = pixels as f32 / foreground.iter().sum::<f32>().max(1.0);
    let target_mean = target_cpu.mean_position();
    let target = BurnTargetExample {
        target_rgb: tensor(render.rgb, [pixels, 3], &device),
        target_density: tensor(render.density, [pixels, 1], &device),
        target_foreground: tensor(foreground, [pixels, 1], &device),
        target_foreground_scale: foreground_scale,
        target_mean: tensor(
            [target_mean[0], target_mean[1]].to_vec(),
            [1, 2],
            &device,
        ),
        target_positions: tensor(
            target_cpu
                .positions
                .iter()
                .flat_map(|position| [position[0], position[1]])
                .collect(),
            [target_cpu.positions.len(), 2],
            &device,
        ),
        pixel_xy: tensor(pixel_xy_values(IMAGE_SIZE), [pixels, 2], &device),
        pixel_size: target_cpu.pixel_size,
        target_points: target_cpu.point_count(),
        particle_count: PARTICLES,
        update_prob: 1.0,
        seed_scale: 0.2,
        target_cpu,
    };
    let positions = (0..BATCHES)
        .flat_map(|_| {
            (0..PARTICLES).flat_map(|particle| {
                let angle = particle as f32 * std::f32::consts::TAU / PARTICLES as f32;
                [0.1 * angle.cos(), 0.1 * angle.sin()]
            })
        })
        .collect::<Vec<_>>();
    let states = vec![0.0; BATCHES * PARTICLES * state_dims];
    let particle_pixel_size = tensor(
        (0..BATCHES)
            .flat_map(|_| {
                (0..PARTICLES)
                    .map(|particle| target.pixel_size * (1.0 + 0.5 * (particle % 3) as f32))
            })
            .collect(),
        [BATCHES, PARTICLES],
        &device,
    );
    let particle_output_scale = tensor(
        vec![target.target_points as f32 / PARTICLES as f32; BATCHES * PARTICLES],
        [BATCHES, PARTICLES],
        &device,
    );
    let x = tensor3(positions, [BATCHES, PARTICLES, 2], &device);
    let s = tensor3(states, [BATCHES, PARTICLES, state_dims], &device);
    let render_mse = tensor1_vec(
        adaptive_uncentered_render_rgb_mse_batch(
            &x,
            &s,
            std::slice::from_ref(&target),
            &indices,
            config,
            particle_pixel_size.clone(),
            particle_output_scale.clone(),
        )
        .inner(),
    )
    .unwrap();

    for backend in [
        Target2dLossBackend::Dense,
        Target2dLossBackend::TiledAdjoint,
    ] {
        let loss = adaptive_target_splat_loss_batch_vector_base_only_selected(
            &x,
            &s,
            std::slice::from_ref(&target),
            &indices,
            DirectBasisTrainConfig {
                target2d_loss_backend: backend,
                ..config
            },
            tensor(
                vec![1.0; BATCHES * PARTICLES],
                [BATCHES, PARTICLES],
                &device,
            ),
            particle_pixel_size.clone(),
            particle_output_scale.clone(),
            Tensor::<BurnBackend, 1>::zeros([BATCHES], &device),
        )
        .unwrap();
        let splat = tensor1_vec(loss.splat.inner()).unwrap();
        assert_eq!(splat.len(), render_mse.len());
        for row in 0..BATCHES {
            assert!(
                (splat[row] - render_mse[row]).abs() < 1.0e-6,
                "{backend:?} render-only adaptive loss diverged at row {row}: splat={} mse={}",
                splat[row],
                render_mse[row],
            );
        }
    }
}

fn train_adaptive_recurrent_smoke(
    model: &mut NpaModel,
    frozen_base: Option<NpaModel>,
    event_training: crate::adaptive::AdaptiveTarget2dEventTrainingConfig,
) -> BurnDenseOracleBatchOutput {
    let residual_perception_semantics = frozen_base.as_ref().map(|_| {
        burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible
    });
    let npa = model.config.clone();
    let target = TargetImage2d {
        source_width: 8,
        source_height: 8,
        positions: vec![[-0.2, 0.1], [0.2, -0.1]],
        colors: vec![[0.2, 0.8, 0.1], [0.8, 0.2, 0.4]],
        pixel_size: 0.25,
        threshold: 0.05,
        aabb: [-1.0, 1.0, -1.0, 1.0],
    };
    let example = DirectBasisExample {
        target,
        adapter: NpaLowRankAdapter::zeros(&npa, 1, 1.0),
        last_train_loss: None,
        particle_count: Some(8),
        update_prob: Some(1.0),
        seed_scale: Some(0.2),
    };
    let mut train = test_direct_config(8);
    train.steps = 1;
    train.example_batch_size = 1;
    train.pool_size = 1;
    train.use_particle_pool = true;
    train.loss_on_final_chunk_only = true;
    train.eval_interval = 1;
    train.loss_config.image_size = 8;
    train.target2d_loss_backend = Target2dLossBackend::Dense;
    let material = crate::adaptive::AdaptiveTarget2dMaterialConfig {
        reference_particle_count: 11,
        total_measure: 11.0,
        fine_bandwidth: 0.1,
        bandwidth_exponent: 0.5,
        max_initial_fine_units: 4,
        seed_layout: crate::adaptive::AdaptiveMaterialSeedLayout::CanonicalGrouped,
        seed_measure_ratio: 1.0,
    }
    .layout(8, 0.01, 0.4)
    .unwrap();
    let mut perception = burn_automata_kernels::AdaptivePerceptionConfig::growing_2d();
    perception.max_neighbors = 8;
    perception.graph_policy = burn_automata_kernels::AdaptiveGraphPolicy::RawSupport;
    let seed_bank = test_adaptive_target2d_seed_bank(1, 8, npa.state_dims, train);
    let output = adaptive_training::train_adaptive_target2d_burn_dense(
        model,
        &example,
        Target2dOracleTrainPlan {
            train,
            steps_per_repetition: 1,
            repetitions: 1,
            optimizer: AdamWConfig {
                learning_rate: 1.0e-3,
                ..AdamWConfig::default()
            },
            scheduler_milestones: Vec::new(),
            scheduler_gamma: 1.0,
        },
        AdaptiveTarget2dBurnConfig {
            material,
            topology: crate::adaptive::AdaptiveTarget2dTopologyConfig {
                enabled: true,
                start_step: 1,
                interval_steps: 1,
                ..crate::adaptive::AdaptiveTarget2dTopologyConfig::default()
            },
            perception,
            perception_options: burn_automata_kernels::AdaptiveNpaPerceptionOptions {
                eps0: npa.eps0,
                scale_equivariance: npa.scale_equivariant(),
                particle_density_equivariance: npa.particle_density_equivariant(),
                log_norm_grad: npa.log_norm_grad,
                log_norm_density_grad: npa.log_norm_density_grad,
                position_features: npa.position_features,
            },
            perception_semantics:
                burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible,
            residual_perception_semantics,
            seed_bank,
            frozen_base,
            material_scale_conditioning: false,
            optimize_material_scale_only: false,
            log1p_trajectory_loss: false,
            trajectory_tail_fraction: 0.0,
            trajectory_tail_weight: 0.0,
            compatible_residual_material_features: false,
            compact_recurrent_memory_dims: 0,
            fresh_seed_trajectories: 1,
            checkpoint_horizons: vec![train.rollout_steps],
            max_pool_age_steps: 0,
            pool_age_strata: 0,
            backward_loss_scale: 1.0,
            event_training,
        },
        None,
        None,
    )
    .unwrap();
    model.validate().unwrap();
    output
}

#[test]
fn adaptive_recurrent_target2d_smoke_updates_the_shared_rule() {
    let mut model = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 91);
    let initial = model.weights.clone();
    let output = train_adaptive_recurrent_smoke(
        &mut model,
        None,
        crate::adaptive::AdaptiveTarget2dEventTrainingConfig::default(),
    );
    assert_eq!(output.history.len(), 1);
    assert!(output.history[0].loss.is_finite());
    assert!(output.history[0].base_grad_norm > 0.0);
    assert_eq!(output.metrics["topology_events"].as_u64(), Some(1));
    let best_step = output.metrics["best_fresh_seed_eval_step"]
        .as_u64()
        .unwrap() as usize;
    if best_step == 0 {
        assert_eq!(model.weights.w1, initial.w1);
    } else {
        assert!(max_abs_difference(&model.weights.w1, &initial.w1) > 0.0);
    }
}

#[test]
fn adaptive_recurrent_target2d_event_training_adds_a_recovery_chunk() {
    let mut model = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 93);
    let output = train_adaptive_recurrent_smoke(
        &mut model,
        None,
        crate::adaptive::AdaptiveTarget2dEventTrainingConfig {
            enabled: true,
            post_event_recovery_steps: 1,
            post_event_loss_weight: 2.0,
            post_event_degradation_weight: 1.0,
            checkpoint_drift_penalty_weight: 1.0,
            min_event_trajectories_per_batch: 1,
            max_recovery_extension_steps: 1,
        },
    );

    assert_eq!(
        output.metrics["event_training"]["scheduled_event_rows"].as_u64(),
        Some(2)
    );
    assert_eq!(
        output.metrics["event_training"]["post_event_row_chunks"].as_u64(),
        Some(1)
    );
    assert_eq!(
        output.metrics["event_training"]["recovery_extension_steps"].as_u64(),
        Some(1)
    );
    assert_eq!(
        output.metrics["event_training"]["quota_shortfall_rows"].as_u64(),
        Some(0)
    );
    assert_eq!(
        output.metrics["event_training"]["quota_met_steps"].as_u64(),
        Some(1)
    );
    assert_eq!(output.metrics["optimizer_step"].as_u64(), Some(1));
    assert_eq!(
        output.metrics["event_training"]["post_event_degradation_weight"].as_f64(),
        Some(1.0)
    );
    assert_eq!(
        output.metrics["event_training"]["checkpoint_drift_penalty_weight"].as_f64(),
        Some(1.0)
    );
    assert!(
        output.metrics["event_training"]["mean_post_event_loss"]
            .as_f64()
            .is_some_and(f64::is_finite)
    );
}

#[test]
fn adaptive_recurrent_target2d_smoke_updates_only_the_compatible_residual() {
    let base = NpaModel::upstream_seeded(NpaConfig::growing_2d(), 91);
    let frozen_weights = base.weights.clone();
    let mut residual = NpaModel {
        config: base.config.clone(),
        weights: NpaWeights::zero_output_seeded(&base.config, 0x6c6f_6361_6c5f_6e70),
    };
    let initial_residual = residual.weights.clone();
    let output = train_adaptive_recurrent_smoke(
        &mut residual,
        Some(base.clone()),
        crate::adaptive::AdaptiveTarget2dEventTrainingConfig::default(),
    );

    assert_eq!(
        output.metrics["training_path"].as_str(),
        Some("adaptive_recurrent_target2d_frozen_base_compatible_residual")
    );
    assert_eq!(base.weights.w1, frozen_weights.w1);
    assert_eq!(base.weights.b1, frozen_weights.b1);
    assert_eq!(base.weights.w2, frozen_weights.w2);
    assert_eq!(base.weights.b2, frozen_weights.b2);
    assert!(output.history[0].base_grad_norm > 0.0);
    let best_step = output.metrics["best_fresh_seed_eval_step"]
        .as_u64()
        .unwrap() as usize;
    if best_step == 0 {
        assert_eq!(residual.weights.w2, initial_residual.w2);
    } else {
        assert!(max_abs_difference(&residual.weights.w2, &initial_residual.w2) > 0.0);
    }
}

#[test]
fn adaptive_paired_topology_conserves_material_and_is_batch_independent() {
    let device = BurnDevice::default();
    let npa = NpaConfig::growing_2d();
    let material = crate::adaptive::AdaptiveTarget2dMaterialConfig {
        reference_particle_count: 11,
        total_measure: 11.0,
        fine_bandwidth: 0.1,
        bandwidth_exponent: 0.5,
        max_initial_fine_units: 4,
        seed_layout: crate::adaptive::AdaptiveMaterialSeedLayout::CanonicalGrouped,
        seed_measure_ratio: 1.0,
    }
    .layout(8, 0.01, 0.4)
    .unwrap();
    assert_eq!(
        material.fine_units,
        Some(vec![4, 1, 1, 1, 1, 1, 1, 1])
    );
    let topology = BurnAdaptiveTopology::new(
        &AdaptiveTarget2dBurnConfig {
            material: material.clone(),
            topology: crate::adaptive::AdaptiveTarget2dTopologyConfig {
                enabled: true,
                start_step: 0,
                end_step: 0,
                split_radius_scale: 1.0,
                merge_detail_scale: 0.0,
                min_relative_gain: 0.0,
                interval_steps: 1,
                events_per_interval: 1,
            },
            perception: burn_automata_kernels::AdaptivePerceptionConfig::growing_2d(),
            perception_options: burn_automata_kernels::AdaptiveNpaPerceptionOptions {
                eps0: npa.eps0,
                scale_equivariance: npa.scale_equivariant(),
                particle_density_equivariance: npa.particle_density_equivariant(),
                log_norm_grad: npa.log_norm_grad,
                log_norm_density_grad: npa.log_norm_density_grad,
                position_features: npa.position_features,
            },
            perception_semantics:
                burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible,
            residual_perception_semantics: None,
            seed_bank: test_adaptive_target2d_seed_bank(
                1,
                material.active_particle_count(),
                npa.state_dims,
                test_direct_config(material.active_particle_count()),
            ),
            frozen_base: None,
            material_scale_conditioning: false,
            optimize_material_scale_only: false,
            log1p_trajectory_loss: false,
            trajectory_tail_fraction: 0.0,
            trajectory_tail_weight: 0.0,
            compatible_residual_material_features: false,
            compact_recurrent_memory_dims: 0,
            fresh_seed_trajectories: 1,
            checkpoint_horizons: vec![1],
            max_pool_age_steps: 0,
            pool_age_strata: 0,
            backward_loss_scale: 1.0,
            event_training: crate::adaptive::AdaptiveTarget2dEventTrainingConfig::default(),
        },
        &device,
    )
    .unwrap();

    let positions = (0..2)
        .flat_map(|batch| {
            (0..8).flat_map(move |row| {
                let order = if batch == 0 { row } else { 7 - row };
                [
                    batch as f32 * 2.0 + order as f32 * 0.1,
                    batch as f32 * -1.5 + order as f32 * 0.04,
                ]
            })
        })
        .collect::<Vec<_>>();
    let states = (0..2)
        .flat_map(|batch| {
            (0..8).flat_map(move |row| {
                [
                    batch as f32 + row as f32 * 0.03,
                    row as f32 * -0.07,
                    batch as f32 * 0.2 + row as f32 * 0.11,
                ]
            })
        })
        .collect::<Vec<_>>();
    let detail = vec![
        10.0, 0.0, 0.2, 0.3, 0.4, 2.0, 3.0, 4.0, 9.0, 4.0, 3.0, 2.0, 0.4, 0.3, 0.2, 0.0,
    ];
    let batch_positions = tensor3(positions.clone(), [2, 8, 2], &device);
    let batch_states = tensor3(states.clone(), [2, 8, 3], &device);
    let (next_positions, next_states, events) =
        topology.apply(batch_positions, batch_states, tensor(detail.clone(), [2, 8], &device));
    assert_eq!(events, 2);
    let next_positions = tensor3_vec(next_positions.inner()).unwrap();
    let next_states = tensor3_vec(next_states.inner()).unwrap();

    let weighted_sum = |values: &[f32], batch: usize, channels: usize| {
        let mut sum = vec![0.0_f32; channels];
        for (row, units) in material.fine_units.as_ref().unwrap().iter().enumerate() {
            for channel in 0..channels {
                sum[channel] +=
                    values[(batch * 8 + row) * channels + channel] * *units as f32;
            }
        }
        sum
    };
    for batch in 0..2 {
        assert!(
            max_abs_difference(
                &weighted_sum(&positions, batch, 2),
                &weighted_sum(&next_positions, batch, 2),
            ) < 2.0e-5
        );
        assert!(
            max_abs_difference(
                &weighted_sum(&states, batch, 3),
                &weighted_sum(&next_states, batch, 3),
            ) < 2.0e-5
        );
    }

    for batch in 0..2 {
        let position_start = batch * 8 * 2;
        let state_start = batch * 8 * 3;
        let (single_positions, single_states, single_events) = topology.apply(
            tensor3(
                positions[position_start..position_start + 16].to_vec(),
                [1, 8, 2],
                &device,
            ),
            tensor3(
                states[state_start..state_start + 24].to_vec(),
                [1, 8, 3],
                &device,
            ),
            tensor(detail[batch * 8..batch * 8 + 8].to_vec(), [1, 8], &device),
        );
        assert_eq!(single_events, 1);
        assert!(
            max_abs_difference(
                &next_positions[position_start..position_start + 16],
                &tensor3_vec(single_positions.inner()).unwrap(),
            ) < 1.0e-6
        );
        assert!(
            max_abs_difference(
                &next_states[state_start..state_start + 24],
                &tensor3_vec(single_states.inner()).unwrap(),
            ) < 1.0e-6
        );
    }
}

#[test]
fn adaptive_continuous_topology_conserves_material_and_is_batch_independent() {
    let device = BurnDevice::default();
    let npa = NpaConfig::growing_2d();
    let material = crate::adaptive::AdaptiveTarget2dMaterialConfig {
        reference_particle_count: 8,
        total_measure: 8.0,
        fine_bandwidth: 0.1,
        bandwidth_exponent: 0.0,
        max_initial_fine_units: 4,
        seed_layout: crate::adaptive::AdaptiveMaterialSeedLayout::GradedContinuous,
        seed_measure_ratio: 1.44,
    }
    .layout(8, 0.01, 0.4)
    .unwrap();
    let mean_measure =
        material.represented_measure.iter().sum::<f32>() / material.active_particle_count() as f32;
    let coarse_rows = material
        .represented_measure
        .iter()
        .enumerate()
        .filter_map(|(row, measure)| (*measure > mean_measure).then_some(row))
        .collect::<Vec<_>>();
    let fine_rows = material
        .represented_measure
        .iter()
        .enumerate()
        .filter_map(|(row, measure)| (*measure < mean_measure).then_some(row))
        .collect::<Vec<_>>();
    assert!(!coarse_rows.is_empty() && !fine_rows.is_empty());

    let topology = BurnAdaptiveTopology::new(
        &AdaptiveTarget2dBurnConfig {
            material: material.clone(),
            topology: crate::adaptive::AdaptiveTarget2dTopologyConfig {
                enabled: true,
                start_step: 0,
                end_step: 2,
                split_radius_scale: 1.0,
                merge_detail_scale: 0.0,
                min_relative_gain: 0.0,
                interval_steps: 1,
                events_per_interval: 2,
            },
            perception: burn_automata_kernels::AdaptivePerceptionConfig::growing_2d(),
            perception_options: burn_automata_kernels::AdaptiveNpaPerceptionOptions {
                eps0: npa.eps0,
                scale_equivariance: npa.scale_equivariant(),
                particle_density_equivariance: npa.particle_density_equivariant(),
                log_norm_grad: npa.log_norm_grad,
                log_norm_density_grad: npa.log_norm_density_grad,
                position_features: npa.position_features,
            },
            perception_semantics:
                burn_automata_kernels::AdaptivePerceptionSemantics::NpaCompatible,
            residual_perception_semantics: None,
            seed_bank: test_adaptive_target2d_seed_bank(
                1,
                material.active_particle_count(),
                npa.state_dims,
                test_direct_config(material.active_particle_count()),
            ),
            frozen_base: None,
            material_scale_conditioning: false,
            optimize_material_scale_only: false,
            log1p_trajectory_loss: false,
            trajectory_tail_fraction: 0.0,
            trajectory_tail_weight: 0.0,
            compatible_residual_material_features: false,
            compact_recurrent_memory_dims: 0,
            fresh_seed_trajectories: 1,
            checkpoint_horizons: vec![1],
            max_pool_age_steps: 0,
            pool_age_strata: 0,
            backward_loss_scale: 1.0,
            event_training: crate::adaptive::AdaptiveTarget2dEventTrainingConfig::default(),
        },
        &device,
    )
    .unwrap();
    assert!(topology.should_apply(0));
    assert!(topology.should_apply(2));
    assert!(!topology.should_apply(3));
    assert_eq!(topology.steps_until_next_event(&[0, 2], 4), 1);
    assert_eq!(
        topology.scheduled_event_delay_steps(&[0, 1], &[1, 2]),
        vec![Some(0), Some(0)]
    );
    assert_eq!(
        topology.scheduled_event_delay_steps(&[0, 1], &[2, 3]),
        vec![Some(1), Some(1)]
    );

    let positions = (0..2)
        .flat_map(|batch| {
            (0..8).flat_map(move |row| {
                [
                    batch as f32 * 1.3 + row as f32 * 0.07,
                    batch as f32 * -0.8
                        + row as f32 * -0.03
                        + (row % 3) as f32 * 0.04,
                ]
            })
        })
        .collect::<Vec<_>>();
    let states = (0..2)
        .flat_map(|batch| {
            (0..8).flat_map(move |row| {
                [
                    batch as f32 + row as f32 * 0.05,
                    row as f32 * -0.09,
                    batch as f32 * 0.3 + row as f32 * 0.13,
                ]
            })
        })
        .collect::<Vec<_>>();
    let mut detail = vec![1.0; 16];
    for batch in 0..2 {
        let offset = batch * 8;
        detail[offset + coarse_rows[0]] = 10.0 - batch as f32;
        detail[offset + coarse_rows[1]] = 8.0 - batch as f32;
        detail[offset + fine_rows[0]] = 0.0;
        detail[offset + fine_rows[1]] = 0.25;
    }
    let (scheduled_positions, scheduled_states, scheduled_events) = topology.apply_scheduled(
        tensor3(positions.clone(), [2, 8, 2], &device),
        tensor3(states.clone(), [2, 8, 3], &device),
        tensor(detail.clone(), [2, 8], &device),
        &[0, 2],
        &[1, 3],
    );
    assert_eq!(
        topology.scheduled_event_rows(&[0, 2], &[1, 3]),
        vec![true, false]
    );
    assert_eq!(scheduled_events, 2);
    let scheduled_positions = tensor3_vec(scheduled_positions.inner()).unwrap();
    let scheduled_states = tensor3_vec(scheduled_states.inner()).unwrap();
    assert!(max_abs_difference(&scheduled_positions[..16], &positions[..16]) > 1.0e-4);
    assert!(max_abs_difference(&scheduled_states[..24], &states[..24]) > 1.0e-4);
    assert!(max_abs_difference(&scheduled_positions[16..], &positions[16..]) < 1.0e-6);
    assert!(max_abs_difference(&scheduled_states[24..], &states[24..]) < 1.0e-6);

    let (next_positions, next_states, events) = topology.apply(
        tensor3(positions.clone(), [2, 8, 2], &device),
        tensor3(states.clone(), [2, 8, 3], &device),
        tensor(detail.clone(), [2, 8], &device),
    );
    assert_eq!(events, 4);
    let next_positions = tensor3_vec(next_positions.inner()).unwrap();
    let next_states = tensor3_vec(next_states.inner()).unwrap();

    let weighted_sum = |values: &[f32], batch: usize, channels: usize| {
        let mut sum = vec![0.0_f32; channels];
        for (row, measure) in material.represented_measure.iter().enumerate() {
            for channel in 0..channels {
                sum[channel] += values[(batch * 8 + row) * channels + channel] * measure;
            }
        }
        sum
    };
    let weighted_second_moment = |values: &[f32], batch: usize| {
        let mut moment = [0.0_f32; 3];
        for (row, measure) in material.represented_measure.iter().enumerate() {
            let x = values[(batch * 8 + row) * 2];
            let y = values[(batch * 8 + row) * 2 + 1];
            moment[0] += measure * x * x;
            moment[1] += measure * x * y;
            moment[2] += measure * y * y;
        }
        moment
    };
    for batch in 0..2 {
        assert!(
            max_abs_difference(
                &weighted_sum(&positions, batch, 2),
                &weighted_sum(&next_positions, batch, 2),
            ) < 2.0e-5
        );
        assert!(
            max_abs_difference(
                &weighted_sum(&states, batch, 3),
                &weighted_sum(&next_states, batch, 3),
            ) < 2.0e-5
        );
        assert!(
            max_abs_difference(
                &weighted_second_moment(&positions, batch),
                &weighted_second_moment(&next_positions, batch),
            ) < 2.0e-5
        );
        let position_start = batch * 16;
        let state_start = batch * 24;
        let (single_positions, single_states, single_events) = topology.apply(
            tensor3(
                positions[position_start..position_start + 16].to_vec(),
                [1, 8, 2],
                &device,
            ),
            tensor3(
                states[state_start..state_start + 24].to_vec(),
                [1, 8, 3],
                &device,
            ),
            tensor(detail[batch * 8..batch * 8 + 8].to_vec(), [1, 8], &device),
        );
        assert_eq!(single_events, 2);
        assert!(
            max_abs_difference(
                &next_positions[position_start..position_start + 16],
                &tensor3_vec(single_positions.inner()).unwrap(),
            ) < 1.0e-6
        );
        assert!(
            max_abs_difference(
                &next_states[state_start..state_start + 24],
                &tensor3_vec(single_states.inner()).unwrap(),
            ) < 1.0e-6
        );
    }
}

#[test]
fn adaptive_continuous_topology_ranking_matches_wgpu_quantization_and_ties() {
    let device = BurnDevice::default();
    let rows = Tensor::<BurnBackend, 1, Int>::from_data(
        TensorData::new(vec![7_i64, 2, 5], [3]),
        &device,
    );
    let detail = tensor(
        vec![
            1.0002, 1.0018, 1.0010, // all quantize to 1.0
            2.0060, 2.0061, 2.0062, // all quantize to 2.0078125
        ],
        [2, 3],
        &device,
    );
    let ranked = stable_local_detail_rank(detail);
    let high = stable_local_detail_score(ranked.clone(), rows.clone(), 8, true)
        .topk_with_indices(3, 1)
        .1
        .inner()
        .into_data()
        .to_vec::<i32>()
        .unwrap();
    let low = stable_local_detail_score(ranked, rows, 8, false)
        .topk_with_indices(3, 1)
        .1
        .inner()
        .into_data()
        .to_vec::<i32>()
        .unwrap();
    // Local row 1 maps to global row 2, then local row 2 maps to row 5.
    assert_eq!(high, vec![1, 2, 0, 1, 2, 0]);
    assert_eq!(low, high);
}

#[test]
fn phase_batch_sampler_covers_all_examples_across_epoch() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut sampler = PhaseBatchSampler::new(10, 3, &mut rng);
    let mut counts = [0usize; 10];
    for _ in 0..4 {
        let batch = sampler.next_batch(&mut rng);
        assert_eq!(batch.len(), 3);
        let mut sorted = batch.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), batch.len());
        for idx in batch {
            counts[idx] += 1;
        }
    }

    assert!(counts.iter().all(|count| *count > 0));
    let stats = sample_update_stats(&counts);
    assert_eq!(stats.zero_update_examples, 0);
    assert_eq!(stats.total_updates, 12);
}

#[test]
fn phase_batch_sampler_full_batch_returns_each_example_once() {
    let mut rng = StdRng::seed_from_u64(11);
    let mut sampler = PhaseBatchSampler::new(8, 0, &mut rng);
    let mut batch = sampler.next_batch(&mut rng);
    batch.sort_unstable();

    assert_eq!(batch, (0..8).collect::<Vec<_>>());
}

#[test]
fn checkpoint_selection_does_not_compare_different_validation_contracts() {
    let frequent = BurnE2eValidationContract {
        examples: 16,
        particles: 2_048,
        horizons: vec![512],
        selection_horizon_min_steps: 512,
    };
    let final_contract = BurnE2eValidationContract {
        examples: 16,
        particles: 4_096,
        horizons: vec![96, 256, 512, 1_024],
        selection_horizon_min_steps: 256,
    };

    assert!(!comparable_selection_score_is_better(
        Some(&frequent),
        15.049,
        Some(&final_contract),
        13.707,
    ));
    assert!(comparable_selection_score_is_better(
        Some(&frequent),
        15.049,
        None,
        -0.185,
    ));
    assert!(!comparable_selection_score_is_better(
        None,
        -0.180,
        Some(&frequent),
        15.049,
    ));
    assert!(comparable_selection_score_is_better(
        Some(&final_contract),
        13.707,
        Some(&final_contract),
        13.236,
    ));
}

#[test]
fn checkpoint_selection_uses_generated_quality_when_schedule_crosses_substrate() {
    assert!(!e2e_schedule_selects_generated_rollout(true, 8_750, 8_750));
    assert!(e2e_schedule_selects_generated_rollout(true, 8_751, 8_750));
    assert!(e2e_schedule_selects_generated_rollout(false, 8_750, 8_750));
}

#[test]
fn terminal_checkpoint_can_be_resumed_for_evaluation_without_an_optimizer_step() {
    super::entrypoints::validate_e2e_resume_completed_step(7_250, 7_250).unwrap();
    assert!(super::entrypoints::validate_e2e_resume_completed_step(7_251, 7_250).is_err());
}

#[test]
fn best_training_checkpoint_keeps_base_when_refine_regresses() {
    let train_phase = test_phase(Some(5.8), 300);
    let train_refine_phase = test_phase(Some(6.1), 0);

    let (loss, step) = best_training_checkpoint(300, &train_phase, &train_refine_phase);

    assert_eq!(loss, Some(5.8));
    assert_eq!(step, 300);
}

#[test]
fn best_training_checkpoint_offsets_better_refine_step() {
    let train_phase = test_phase(Some(5.8), 300);
    let train_refine_phase = test_phase(Some(4.9), 120);

    let (loss, step) = best_training_checkpoint(300, &train_phase, &train_refine_phase);

    assert_eq!(loss, Some(4.9));
    assert_eq!(step, 420);
}

fn test_phase(best_loss: Option<f32>, best_step: usize) -> BurnPhaseReport {
    BurnPhaseReport {
        history: Vec::new(),
        best_loss,
        best_step,
        best_geometry_score: None,
        sample_updates: sample_update_stats(&[]),
    }
}
