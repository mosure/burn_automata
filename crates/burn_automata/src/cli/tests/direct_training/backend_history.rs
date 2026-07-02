use super::*;

#[test]
fn render_direct_rollout_backend_applies_mlp_gradients() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let mut model =
        local_growth_student_model(config, 19, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let before = model.weights.w2.clone();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.72);
    let cfg = RenderProxyTrainingConfig {
        particles: 32,
        rollout_steps: 2,
        gradient_particles: 32,
        perception_position_gain: 1.0,
        coverage_samples: 0,
        direct_line_search: true,
        direct_line_search_scales: vec![0.5, 1.0, 2.0],
        seed_scale: 0.72,
        seed_mode: ParticleSeed::UniformCircle,
        render: RenderLossConfig {
            image_size: 8,
            target_samples: 128,
            world_scale: 1.44,
            ..RenderLossConfig::default()
        },
        sgd: SgdConfig {
            learning_rate: 1.0e-3,
            grad_clip_norm: 1.0,
            weight_decay: 0.0,
        },
        ..direct_rollout_test_config()
    };
    let (trace, trajectory) = render_training_trajectory(&model, &grid, &cfg, 0).unwrap();
    let gradient = RenderProxyGradientRows {
        row_indices: (0..cfg.particles).collect(),
        gradients: vec![[1.0, 0.25, -0.5]; cfg.particles],
        opacity_gradients: vec![0.0; cfg.particles],
        scale_gradients: vec![0.0; cfg.particles],
        color_gradients: vec![[0.1, -0.2, 0.05]; cfg.particles],
    };
    let report = render_direct_rollout_training_step(
        &mut model,
        &grid,
        &target,
        &trace,
        &trajectory,
        &gradient,
        &cfg,
        cfg.seed,
    )
    .unwrap();

    assert_eq!(report.rows, cfg.particles * cfg.rollout_steps);
    assert!(report.initial_loss.is_finite());
    assert!(report.final_loss.is_finite());
    assert_eq!(report.best_loss, report.initial_loss.min(report.final_loss));
    assert_eq!(report.history[0].loss, report.final_loss);
    assert!(report.history[0].grad_norm.is_finite());
    assert!(report.history[0].grad_norm > 0.0);
    assert_ne!(model.weights.w2, before);
}
#[test]
fn render_proxy_history_records_direct_line_search_candidates() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let mut model =
        local_growth_student_model(config, 21, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.72);
    let cfg = RenderProxyTrainingConfig {
        particles: 32,
        rollout_steps: 2,
        gradient_particles: 32,
        perception_position_gain: 1.0,
        direct_line_search: true,
        direct_line_search_scales: vec![0.5, 1.0, 2.0],
        seed: 17,
        seed_scale: 0.72,
        seed_mode: ParticleSeed::UniformCircle,
        render: RenderLossConfig {
            image_size: 8,
            target_samples: 64,
            world_scale: 1.44,
            ..RenderLossConfig::default()
        },
        sgd: SgdConfig {
            learning_rate: 1.0e-3,
            grad_clip_norm: 1.0,
            weight_decay: 0.0,
        },
        ..direct_rollout_test_config()
    };

    let report = run_render_proxy_training(&mut model, &grid, &target, cfg).unwrap();
    let history = &report.history[0];
    let candidates = &history.direct_line_search_candidates;

    assert!(candidates.len() >= 3);
    let candidate_scales = candidates
        .iter()
        .map(|candidate| candidate.scale)
        .collect::<Vec<_>>();
    for scale in [0.5, 1.0, 2.0] {
        assert!(
            candidate_scales.contains(&scale),
            "line search should retain requested scale {scale}"
        );
    }
    assert!(candidates.iter().all(|candidate| candidate.inner_step == 0));
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.render_loss.is_finite())
    );
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.score.is_finite())
    );
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.train_final_loss.is_finite())
    );
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.train_grad_norm.is_finite())
    );
    let selected_count = candidates
        .iter()
        .filter(|candidate| candidate.selected_checkpoint || candidate.selected_progress)
        .count();
    assert!(selected_count <= 1);
    let serialized = serde_json::to_value(history).unwrap();
    assert_eq!(
        serialized["direct_line_search_candidates"]
            .as_array()
            .unwrap()
            .len(),
        candidates.len()
    );
}
#[test]
fn adaptive_line_search_refines_underactive_to_bursty_scale_gap() {
    let reports = vec![
        adaptive_line_search_candidate(8.0, 47, 0.325, 0.08),
        adaptive_line_search_candidate(16.0, 113, 0.875, 0.274),
    ];

    let scales = adaptive_direct_line_search_refinement_scales(&reports, 128);

    assert_eq!(scales.len(), 3);
    assert!(scales.iter().all(|scale| *scale > 8.0 && *scale < 16.0));
    assert!(
        scales.windows(2).all(|pair| pair[0] < pair[1]),
        "refinement scales should preserve log-space order"
    );
    assert!((scales[1] - (8.0_f32 * 16.0).sqrt()).abs() <= 1.0e-5);
}
#[test]
fn adaptive_line_search_skips_pairs_without_activation_bracket() {
    let reports = vec![
        adaptive_line_search_candidate(4.0, 36, 0.233, 0.10),
        adaptive_line_search_candidate(8.0, 47, 0.325, 0.08),
    ];

    assert!(adaptive_direct_line_search_refinement_scales(&reports, 128).is_empty());
}
#[test]
fn direct_rollout_objective_diagnostics_reports_channel_pressure() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = local_growth_student_model(config, 31, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let cfg = RenderProxyTrainingConfig {
        particles: 32,
        rollout_steps: 4,
        gradient_particles: 8,
        motion_gain: 0.0,
        perception_position_gain: 0.0,
        trajectory_mesh_gain: ROBUST_3D_TRAJECTORY_MESH_GAIN,
        liveness_gain: 1.0,
        liveness_update_multiplier: 20.0,
        coverage_gain: ROBUST_3D_COVERAGE_GAIN,
        coverage_samples: 32,
        extent_gain: ROBUST_3D_EXTENT_GAIN,
        surface_gain: ROBUST_3D_SURFACE_GAIN,
        surface_escape_gain: ROBUST_3D_SURFACE_ESCAPE_GAIN,
        opacity_gain: ROBUST_3D_OPACITY_GAIN,
        material_liveness_gain: ROBUST_3D_MATERIAL_LIVENESS_GAIN,
        direct_output_gradient_rms_cap: 0.125,
        seed: 37,
        seed_mode: ParticleSeed::TorusLocalSubstrateGrowth3d,
        sgd: SgdConfig {
            learning_rate: 1.0e-3,
            grad_clip_norm: 1.0,
            weight_decay: 0.0,
        },
        ..direct_rollout_test_config()
    };
    let (_, trajectory) = render_training_trajectory(&model, &grid, &cfg, 0).unwrap();
    let diagnostics = direct_rollout_objective_diagnostics(&model, &target, &trajectory, &cfg)
        .expect("diagnostics should evaluate direct objective pressure");

    assert_eq!(diagnostics.snapshots, cfg.rollout_steps);
    assert_eq!(diagnostics.rows, cfg.particles * cfg.rollout_steps);
    assert!(
        diagnostics.temporal_liveness_rms > 0.0,
        "temporal liveness objective should be visible before MLP backprop"
    );
    assert!(
        diagnostics.terminal_liveness_state_rms > 0.0,
        "terminal liveness anchor should be visible in state-adjoint diagnostics"
    );
    assert!(diagnostics.terminal_liveness_state_nonzero_fraction > 0.0);
    assert!(
        diagnostics.mesh_motion_liveness_rms > 0.0,
        "mesh-motion liveness objective should activate moving local-front rows"
    );
    assert!(diagnostics.mesh_motion_liveness_nonzero_fraction > 0.0);
    assert!(
        diagnostics.target_coverage_liveness_rms > 0.0,
        "target-coverage liveness objective should activate coverage-critical local-front rows"
    );
    assert!(diagnostics.target_coverage_liveness_nonzero_fraction > 0.0);
    assert!(
        diagnostics.material_coverage_liveness_rms > 0.0,
        "material-coverage liveness objective should activate local-front rows that improve visible support"
    );
    assert!(diagnostics.material_coverage_liveness_nonzero_fraction > 0.0);
    assert!(
        diagnostics.phase_rms > 0.0,
        "phase objective should be visible before MLP backprop"
    );
    assert!(
        diagnostics.liveness_phase_memory_rms > 0.0,
        "liveness pressure should also supervise recurrent phase-memory outputs"
    );
    assert!(diagnostics.liveness_phase_memory_nonzero_fraction > 0.0);
    assert!(
        diagnostics.mesh_motion_rms > 0.0,
        "trajectory mesh objective should produce motion-channel pressure"
    );
    assert!(
        diagnostics.extent_front_motion_rms > 0.0,
        "extent-front objective should produce outward dormant-front motion pressure"
    );
    assert!(diagnostics.extent_front_motion_nonzero_fraction > 0.0);
    assert!(
        diagnostics.temporal_extent_motion_rms > 0.0,
        "temporal extent objective should produce scheduled expansion pressure"
    );
    assert!(diagnostics.temporal_extent_motion_nonzero_fraction > 0.0);
    assert!(
        diagnostics.extent_motion_memory_rms > 0.0,
        "extent-front motion should also supervise recurrent velocity-memory outputs"
    );
    assert!(diagnostics.extent_motion_memory_nonzero_fraction > 0.0);
    assert!(
        diagnostics.material_coverage_motion_rms > 0.0,
        "material-coverage motion objective should produce visible-support motion pressure"
    );
    assert!(diagnostics.material_coverage_motion_nonzero_fraction > 0.0);
    assert!(
        diagnostics.material_surface_motion_rms > 0.0,
        "material visible-surface objective should produce active/front surface motion pressure"
    );
    assert!(diagnostics.material_surface_motion_nonzero_fraction > 0.0);
    assert!(
        diagnostics.residual_velocity_rms > 0.0,
        "mesh residual objective should supervise velocity state outputs directly"
    );
    assert!(diagnostics.residual_velocity_nonzero_fraction > 0.0);
    assert!(
        diagnostics.motion_memory_rms > 0.0,
        "trajectory mesh objective should also supervise velocity-memory state outputs"
    );
    assert!(
        diagnostics.material_coverage_motion_memory_rms > 0.0,
        "material-coverage motion should also supervise recurrent velocity-memory outputs"
    );
    assert!(diagnostics.material_coverage_motion_memory_nonzero_fraction > 0.0);
    assert!(
        diagnostics.material_coverage_materialization_rms > 0.0,
        "material-coverage candidates should train render-material outputs directly"
    );
    assert!(diagnostics.material_coverage_materialization_nonzero_fraction > 0.0);
    assert!(
        diagnostics.temporal_materialization_rms > 0.0,
        "temporal materialization should train material outputs for scheduled local-front growth"
    );
    assert!(diagnostics.temporal_materialization_nonzero_fraction > 0.0);
    assert!(
        diagnostics.active_surface_materialization_rms > 0.0,
        "active surface materialization should train near-surface active material outputs"
    );
    assert!(diagnostics.active_surface_materialization_nonzero_fraction > 0.0);
    assert!(
        diagnostics.material_visibility_rms > 0.0,
        "material visibility objective should produce material-channel pressure"
    );
    assert!(diagnostics.combined_pre_cap_rms >= diagnostics.combined_post_cap_rms);
    assert!(diagnostics.mesh_motion_post_cap_rms > 0.0);
    assert!(diagnostics.mesh_motion_post_cap_nonzero_fraction > 0.0);
    assert!(
        diagnostics.mesh_motion_post_cap_rms >= cfg.direct_output_gradient_rms_cap * 0.5,
        "combined direct gradients should preserve spatial motion pressure instead of letting liveness/material channels dominate"
    );
    assert!(diagnostics.residual_velocity_post_cap_rms > 0.0);
    assert!(diagnostics.residual_velocity_post_cap_nonzero_fraction > 0.0);
    assert!(diagnostics.motion_memory_post_cap_rms > 0.0);
    assert!(diagnostics.motion_memory_post_cap_nonzero_fraction > 0.0);
    assert!(diagnostics.liveness_post_cap_rms > 0.0);
    assert!(diagnostics.phase_post_cap_rms > 0.0);
    assert!(diagnostics.material_post_cap_rms > 0.0);
}
#[test]
fn direct_rollout_training_honors_supervised_steps_per_round() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let mut model =
        local_growth_student_model(config, 23, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let before = model.weights.w2.clone();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let cfg = RenderProxyTrainingConfig {
        supervised_steps_per_round: 3,
        seed: 29,
        ..direct_rollout_test_config()
    };
    let baseline = render_selection_baseline(&model, &grid, &target, &cfg, cfg.render).unwrap();
    let (report, step_scale, line_search_candidates) = render_direct_rollout_training_steps(
        &mut model, &grid, &target, &cfg, 0, cfg.render, &baseline,
    )
    .unwrap();

    assert_eq!(step_scale, 1.0);
    assert!(line_search_candidates.is_empty());
    assert_eq!(report.steps, cfg.supervised_steps_per_round);
    assert_eq!(report.history.len(), cfg.supervised_steps_per_round);
    assert_eq!(
        report.rows,
        cfg.supervised_steps_per_round * cfg.particles * cfg.rollout_steps
    );
    assert_eq!(report.history[0].step, 1);
    assert_eq!(report.history[2].step, 3);
    assert!(report.initial_loss.is_finite());
    assert!(report.final_loss.is_finite());
    assert!(report.best_loss.is_finite());
    assert!(report.history.iter().all(|entry| entry.loss.is_finite()));
    assert_ne!(model.weights.w2, before);
}
#[test]
fn direct_multiseed_loss_weights_emphasize_worse_rollout_seeds() {
    let weights = direct_rollout_multiseed_loss_weights(&[1.0, 3.0, 1.0]);

    assert!((weights.iter().sum::<f32>() - 1.0).abs() <= 1.0e-6);
    assert!(weights[1] > weights[0]);
    assert!((weights[0] - weights[2]).abs() <= 1.0e-6);

    let uniform = direct_rollout_multiseed_loss_weights(&[2.0, 2.0]);
    assert_eq!(uniform, vec![0.5, 0.5]);

    let invalid = direct_rollout_multiseed_loss_weights(&[f32::NAN, f32::INFINITY]);
    assert_eq!(invalid, vec![0.5, 0.5]);

    let dynamics_weights =
        direct_rollout_multiseed_objective_weights(&[1.0, 1.0, 1.0], &[1.0, 9.0, 1.0]);
    assert!((dynamics_weights.iter().sum::<f32>() - 1.0).abs() <= 1.0e-6);
    assert!(
        dynamics_weights[1] > dynamics_weights[0],
        "dynamic rollout score should upweight morphogenesis-hard seeds even when terminal render loss ties"
    );

    let fallback_weights = direct_rollout_multiseed_objective_weights(&[1.0, 3.0], &[1.0]);
    assert_eq!(
        fallback_weights,
        direct_rollout_multiseed_loss_weights(&[1.0, 3.0])
    );
}
#[test]
fn direct_multiseed_training_reports_actual_weighted_model_loss() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let mut model =
        local_growth_student_model(config, 27, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let before = model.weights.w2.clone();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let cfg = RenderProxyTrainingConfig {
        rollout_steps: 2,
        gradient_particles: 8,
        trajectory_mesh_gain: 0.05,
        trajectory_render_samples: 2,
        coverage_gain: 0.1,
        coverage_samples: 32,
        coverage_normal_weight: 0.1,
        full_coverage_adjoint: true,
        surface_gain: 0.1,
        direct_selection_seed_training: true,
        seed: 31,
        selection_seed: Some(37),
        selection_seeds: vec![43],
        ..direct_rollout_test_config()
    };
    let seeds = render_direct_rollout_training_seeds(&cfg, 0);
    let (trace, trajectory) = render_training_trajectory(&model, &grid, &cfg, 0).unwrap();
    let gradient = render_position_gradient(&trace, &target, cfg.render, &cfg).unwrap();
    let initial_losses = seeds
        .iter()
        .map(|seed| {
            let trace = render_training_trace_for_seed(&model, &grid, &cfg, *seed).unwrap();
            mesh_multiview_render_loss_from_trace(&trace, &target, cfg.render)
                .unwrap()
                .total_loss
        })
        .collect::<Vec<_>>();
    let selection_scores = seeds
        .iter()
        .map(|seed| {
            let case =
                render_selection_case_metrics(&model, &grid, &target, &cfg, cfg.render, *seed)
                    .unwrap();
            render_selection_case_score_with_baseline(*seed, &case, None).score
        })
        .collect::<Vec<_>>();
    let seed_weights =
        direct_rollout_multiseed_objective_weights(&initial_losses, &selection_scores);

    let report = render_direct_rollout_multiseed_training_step(
        &mut model,
        &grid,
        &target,
        &cfg,
        0,
        &trace,
        &trajectory,
        &gradient,
    )
    .unwrap();
    let expected_initial_loss = initial_losses
        .iter()
        .zip(seed_weights.iter())
        .map(|(loss, weight)| loss * weight)
        .sum::<f32>();
    let actual_final_loss = render_direct_rollout_weighted_loss_for_seeds(
        &model,
        &grid,
        &target,
        &cfg,
        cfg.render,
        &seeds,
        &seed_weights,
    )
    .unwrap();

    assert_eq!(seeds.len(), 3);
    assert_eq!(report.steps, 1);
    assert_eq!(report.history.len(), 1);
    assert!(report.initial_loss.is_finite());
    assert!(report.final_loss.is_finite());
    assert!(
        (report.initial_loss - expected_initial_loss).abs() <= 1.0e-6,
        "multiseed report should use the render-loss and rollout-score weighted robust objective"
    );
    assert!(
        (report.final_loss - actual_final_loss).abs() <= 1.0e-6,
        "multiseed report should evaluate the weighted model objective that is kept"
    );
    assert_eq!(report.history[0].loss, report.final_loss);
    assert_eq!(report.best_loss, report.initial_loss.min(report.final_loss));
    assert_ne!(model.weights.w2, before);
}
#[test]
fn render_proxy_history_records_direct_rollout_inner_steps() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let mut model =
        local_growth_student_model(config, 31, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let cfg = RenderProxyTrainingConfig {
        supervised_steps_per_round: 3,
        seed: 37,
        ..direct_rollout_test_config()
    };

    let report = run_render_proxy_training(&mut model, &grid, &target, cfg).unwrap();
    let round = &report.history[0];

    assert_eq!(report.history.len(), 1);
    assert_eq!(round.train_step_count, report.supervised_steps_per_round);
    assert_eq!(
        round.train_loss_history.len(),
        report.supervised_steps_per_round
    );
    assert_eq!(
        round.train_grad_norm_history.len(),
        report.supervised_steps_per_round
    );
    assert_eq!(
        round.train_grad_scale_history.len(),
        report.supervised_steps_per_round
    );
    assert_eq!(round.supervised_loss, round.train_final_loss);
    assert_eq!(
        round.train_loss_history.last().copied().unwrap(),
        round.train_final_loss
    );
    assert_eq!(
        round.train_grad_norm_history.last().copied().unwrap(),
        round.train_grad_norm
    );
    assert_eq!(
        round.train_grad_scale_history.last().copied().unwrap(),
        round.train_grad_scale
    );
    assert!(round.train_loss_history.iter().all(|loss| loss.is_finite()));
    assert!(round.before_selection_loss.is_finite());
    assert!(round.before_selection_score.is_finite());
    assert!(round.before_selection_density_psnr_db.is_finite());
    assert!(round.selection_loss.is_finite());
    assert!(round.selection_score.is_finite());
    assert!(round.selection_density_psnr_db.is_finite());
    assert!(round.train_phase_output_delta_norm.is_finite());
    assert!(round.selection_min_final_active_count <= report.final_gaussian_volume.particles);
    assert!(round.selection_min_newly_activated_fraction.is_finite());
    assert!((0.0..=1.0).contains(&round.selection_min_newly_activated_fraction));
    assert!(
        round
            .selection_min_front_local_newly_activated_fraction
            .is_finite()
    );
    assert!((0.0..=1.0).contains(&round.selection_min_front_local_newly_activated_fraction));
    assert!(
        round.selection_worst_failure_reasons.is_empty()
            || round
                .selection_worst_failure_reasons
                .iter()
                .all(|reason| !reason.is_empty())
    );
}
#[test]
fn render_proxy_training_rolls_back_rejected_round_before_next_round() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let mut model =
        local_growth_student_model(config, 41, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let initial_model = model.clone();
    let target = mesh_target_for_arg(MeshTargetArg::Torus, 0.54);
    let cfg = RenderProxyTrainingConfig {
        rounds: 2,
        seed: 43,
        sgd: SgdConfig {
            learning_rate: 1000.0,
            grad_clip_norm: 100.0,
            weight_decay: 0.0,
        },
        ..direct_rollout_test_config()
    };
    let expected_round_one_trace = render_training_trace(&initial_model, &grid, &cfg, 1).unwrap();
    let expected_round_one_loss =
        mesh_multiview_render_loss_from_trace(&expected_round_one_trace, &target, cfg.render)
            .unwrap()
            .total_loss;

    let report = run_render_proxy_training(&mut model, &grid, &target, cfg).unwrap();

    assert_eq!(report.history.len(), 2);
    assert!(
        report.history[0].rolled_back_to_best_checkpoint,
        "aggressive rejected first round should be rolled back before continuing"
    );
    assert!(!report.history[0].selected_checkpoint);
    assert_eq!(
        report.history[0].train_step_scale, 0.0,
        "rolled-back rounds should report the applied checkpoint scale, not the attempted update scale"
    );
    assert!(
        (report.history[1].before_loss - expected_round_one_loss).abs() <= 1.0e-6,
        "round two should start from the best checkpoint, not the rejected round-one weights"
    );
}
