use super::*;

mod backend_history;
mod gradient_flow;
mod surface_material;
mod validation;

fn adaptive_line_search_candidate(
    scale: f32,
    min_final_active_count: usize,
    min_newly_activated_fraction: f32,
    max_temporal_activation_schedule_error: f32,
) -> DirectLineSearchCandidateReport {
    DirectLineSearchCandidateReport {
        inner_step: 0,
        candidate_kind: "sgd-scale",
        scale,
        material_opacity_bias: 0.0,
        checkpoint_candidate: false,
        progress_candidate: false,
        selected_checkpoint: false,
        selected_progress: false,
        render_loss: 1.0,
        max_render_loss: 1.0,
        score: 1.0,
        density_psnr_db: 0.0,
        min_density_psnr_db: 0.0,
        morphology_non_regressed: true,
        active_surface_max: 0.0,
        target_coverage_fraction: 0.0,
        material_visible_target_mean_distance: 0.0,
        material_visible_target_max_distance: 0.0,
        material_visible_target_coverage_fraction: 0.0,
        strict_surface_active_count: 0,
        strict_surface_materialized_fraction: 0.0,
        strict_surface_material_mean_opacity: f32::NEG_INFINITY,
        strict_surface_material_visible_margin: f32::MAX,
        strict_surface_material_max_visible_margin: f32::MAX,
        material_visible_inactive_fraction: 0.0,
        material_visible_max_inactive_opacity: 0.0,
        material_active_mean_opacity: 0.0,
        material_visible_count: 0,
        active_color_state_mean_abs: 0.0,
        active_color_state_max_abs: 0.0,
        active_color_state_stddev_mean: 0.0,
        surface_covered_bin_fraction: 0.0,
        surface_mean_bin_covered_fraction: 0.0,
        material_visible_surface_covered_bin_fraction: 0.0,
        material_visible_surface_mean_bin_covered_fraction: 0.0,
        surface_normal_covered_bin_fraction: 0.0,
        surface_normal_mean_bin_covered_fraction: 0.0,
        material_visible_surface_normal_covered_bin_fraction: 0.0,
        material_visible_surface_normal_mean_bin_covered_fraction: 0.0,
        material_visible_surface_tail_p99_distance: 0.0,
        material_visible_surface_tail_over_threshold_fraction: 0.0,
        max_dormant_drift_fraction: 0.0,
        max_dormant_drift: 0.0,
        all_dormant_drift_bounded: true,
        min_active_extent_bbox_ratio: 0.0,
        min_active_extent_min_axis_ratio: 0.0,
        min_final_active_count,
        min_newly_activated_fraction,
        min_front_local_newly_activated_fraction: 1.0,
        max_front_liveness_margin: 0.0,
        min_front_liveness_candidate_count: 0,
        max_extent_front_liveness_margin: 0.0,
        min_extent_front_liveness_candidate_count: 0,
        max_temporal_front_liveness_margin: 0.0,
        min_temporal_front_liveness_candidate_count: 0,
        max_temporal_extent_front_liveness_margin: 0.0,
        min_temporal_extent_front_liveness_candidate_count: 0,
        max_temporal_activation_schedule_error,
        all_temporal_activation_progressive: false,
        all_temporal_geometry_progressive: false,
        train_final_loss: 1.0,
        train_grad_norm: 0.0,
        train_grad_scale: 1.0,
        failure_reasons: Vec::new(),
    }
}

fn direct_line_search_test_config() -> RenderProxyTrainingConfig {
    RenderProxyTrainingConfig {
        target: MeshTargetArg::Torus,
        rounds: 1,
        supervised_steps_per_round: 1,
        particles: 128,
        rollout_steps: 8,
        gradient_particles: 32,
        gradient_mode: RenderGradientModeArg::Analytic,
        finite_diff_eps: 1.0e-3,
        motion_gain: 0.0,
        perception_position_gain: 0.0,
        max_update_norm: 0.05,
        trajectory_supervision: true,
        trajectory_render_gain: ROBUST_3D_TRAJECTORY_RENDER_GAIN,
        trajectory_mesh_gain: ROBUST_3D_TRAJECTORY_MESH_GAIN,
        trajectory_render_samples: ROBUST_3D_TRAJECTORY_RENDER_SAMPLES,
        liveness_gain: ROBUST_3D_LIVENESS_GAIN,
        liveness_front_radius: ROBUST_3D_LIVENESS_FRONT_RADIUS,
        liveness_update_multiplier: ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER,
        coverage_gain: ROBUST_3D_COVERAGE_GAIN,
        coverage_samples: ROBUST_3D_COVERAGE_SAMPLES,
        coverage_mode: CoverageUpdateModeArg::SlicedOt,
        coverage_softness: 0.0,
        coverage_repulsion_gain: ROBUST_3D_COVERAGE_REPULSION_GAIN,
        coverage_gap_gain: ROBUST_3D_COVERAGE_REPULSION_GAIN,
        coverage_repulsion_radius: 0.0,
        coverage_normal_weight: ROBUST_3D_COVERAGE_NORMAL_WEIGHT,
        extent_gain: ROBUST_3D_EXTENT_GAIN,
        full_coverage_adjoint: true,
        surface_gain: ROBUST_3D_SURFACE_GAIN,
        surface_escape_gain: ROBUST_3D_SURFACE_ESCAPE_GAIN,
        opacity_gain: ROBUST_3D_OPACITY_GAIN,
        material_liveness_gain: ROBUST_3D_MATERIAL_LIVENESS_GAIN,
        material_tail_gain: ROBUST_3D_MATERIAL_TAIL_GAIN,
        material_suppression_update_multiplier: ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER,
        material_max_opacity_update: ROBUST_3D_MATERIAL_MAX_OPACITY_UPDATE,
        scale_gain: ROBUST_3D_SCALE_GAIN,
        scale_budget_weight: ROBUST_3D_SCALE_BUDGET_WEIGHT,
        max_opacity_update: 0.05,
        direct_output_gradient_rms_cap: ROBUST_3D_DIRECT_OUTPUT_GRADIENT_RMS_CAP,
        direct_line_search: true,
        direct_line_search_scales: vec![1.0],
        direct_material_output_only: false,
        training_backend: RenderTrainingBackendArg::DirectRollout,
        weight_update_mode: RenderWeightUpdateModeArg::Full,
        adapter_rank: 8,
        adapter_alpha: 8.0,
        adapter_seed: 0x00ad_a973,
        direct_selection_seed_training: true,
        seed: 13,
        selection_seed: Some(17),
        selection_seeds: vec![19],
        seed_scale: 0.72,
        seed_mode: ParticleSeed::TorusGrowth3d,
        render: RenderLossConfig::default(),
        sgd: SgdConfig {
            learning_rate: 1.0e-4,
            grad_clip_norm: 1.0,
            weight_decay: 0.0,
        },
    }
}

#[test]
fn material_opacity_bias_line_search_candidates_follow_strict_margin() {
    let cfg = direct_line_search_test_config();
    let mut selection = render_selection_metrics_with_liveness(99.5, 0.68, 1.94, 0.0);
    selection.strict_surface_active_count = 15;
    selection.strict_surface_materialized_fraction = 0.0;
    selection.strict_surface_material_visible_margin = 1.8;

    let candidates = material_opacity_bias_line_search_candidates(&selection, &cfg);

    assert_eq!(candidates.len(), 3);
    assert!(candidates.iter().all(|candidate| *candidate > 0.0));
    assert!(candidates.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(
        candidates[2] <= cfg.material_max_opacity_update * 0.25 + 1.0e-6,
        "material bias line search should remain bounded by a conservative fraction of the material update cap"
    );

    selection.strict_surface_materialized_fraction = 1.0;
    assert!(material_opacity_bias_line_search_candidates(&selection, &cfg).is_empty());
    selection.strict_surface_materialized_fraction = 0.0;
    selection.strict_surface_active_count = 0;
    assert!(material_opacity_bias_line_search_candidates(&selection, &cfg).is_empty());
}
