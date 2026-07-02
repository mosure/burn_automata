use super::*;

pub(crate) fn run_render_proxy_training(
    model: &mut NpaModel,
    grid: &crate::kernels::HashGridConfig,
    target: &TriangleMeshTarget,
    cfg: RenderProxyTrainingConfig,
) -> Result<RenderProxyTrainingReport, Box<dyn std::error::Error>> {
    if cfg.rounds == 0 || cfg.supervised_steps_per_round == 0 {
        return Err(std::io::Error::other(
            "render-proxy training requires non-zero rounds and supervised steps",
        )
        .into());
    }
    if !cfg.finite_diff_eps.is_finite() || cfg.finite_diff_eps <= 0.0 {
        return Err(std::io::Error::other("finite_diff_eps must be positive and finite").into());
    }
    let mut render_cfg = cfg.render;
    if render_cfg.target_samples == 0 {
        render_cfg.target_samples = cfg.particles;
    }
    let initial_trace = render_training_trace(model, grid, &cfg, 0)?;
    let initial_render_loss =
        mesh_multiview_render_loss_from_trace(&initial_trace, target, render_cfg)?;
    let initial_gaussian_volume = gaussian_volume_stats_for_trace(&initial_trace, render_cfg);
    let selection_baseline = render_selection_baseline(model, grid, target, &cfg, render_cfg)?;
    let initial_selection = render_selection_metrics(
        model,
        grid,
        target,
        &cfg,
        render_cfg,
        Some(&selection_baseline),
    )?;
    let mut best_model = model.clone();
    let mut best_selection = initial_selection.clone();
    let mut selected_round = None;
    let mut history = Vec::with_capacity(cfg.rounds);

    for round in 0..cfg.rounds {
        let needs_trajectory = cfg.trajectory_supervision
            || cfg.training_backend == RenderTrainingBackendArg::DirectRollout;
        let (trace, trajectory) = if needs_trajectory {
            render_training_trajectory(model, grid, &cfg, round)?
        } else {
            (render_training_trace(model, grid, &cfg, round)?, Vec::new())
        };
        let before = mesh_multiview_render_loss_from_trace(&trace, target, render_cfg)?;
        let before_selection = render_selection_metrics(
            model,
            grid,
            target,
            &cfg,
            render_cfg,
            Some(&selection_baseline),
        )?;
        let gradient = render_position_gradient(&trace, target, render_cfg, &cfg)?;
        let gradient_rms = (gradient
            .gradients
            .iter()
            .map(|g| g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
            .sum::<f32>()
            / gradient.gradients.len().max(1) as f32)
            .sqrt();
        let opacity_gradient_rms = (gradient
            .opacity_gradients
            .iter()
            .map(|gradient| gradient * gradient)
            .sum::<f32>()
            / gradient.opacity_gradients.len().max(1) as f32)
            .sqrt();
        let scale_gradient_rms = (gradient
            .scale_gradients
            .iter()
            .map(|gradient| gradient * gradient)
            .sum::<f32>()
            / gradient.scale_gradients.len().max(1) as f32)
            .sqrt();
        let direct_objective_diagnostics =
            if cfg.training_backend == RenderTrainingBackendArg::DirectRollout {
                direct_rollout_objective_diagnostics(model, target, &trajectory, &cfg)?
            } else {
                DirectRolloutObjectiveDiagnostics::default()
            };
        let before_training_weights = model.weights.clone();
        let (train_report, train_step_scale, direct_line_search_candidates) =
            match cfg.training_backend {
                RenderTrainingBackendArg::Proxy => {
                    let batch = render_proxy_supervised_batch(
                        model,
                        grid,
                        target,
                        &trace,
                        &trajectory,
                        &gradient,
                        &cfg,
                    )?;
                    let report = run_supervised_training(
                        model,
                        &batch,
                        TrainingRunConfig {
                            steps: cfg.supervised_steps_per_round,
                            report_interval: cfg.supervised_steps_per_round,
                            sgd: cfg.sgd,
                        },
                    )?;
                    (report, 1.0, Vec::new())
                }
                RenderTrainingBackendArg::DirectRollout => render_direct_rollout_training_steps(
                    model,
                    grid,
                    target,
                    &cfg,
                    round,
                    render_cfg,
                    &selection_baseline,
                )?,
            };
        let train_liveness_output_delta_norm = output_channel_delta_norm(
            &before_training_weights,
            &model.weights,
            model.config.hidden_dims,
            model.config.spatial_dims + GROWTH_3D_LIVENESS_CHANNEL,
        );
        let train_phase_output_delta_norm = growth_3d_phase_channel(model.config.state_dims)
            .map(|channel| {
                output_channel_delta_norm(
                    &before_training_weights,
                    &model.weights,
                    model.config.hidden_dims,
                    model.config.spatial_dims + channel,
                )
            })
            .unwrap_or(0.0);
        let train_motion_output_delta_norm = spatial_output_delta_norm(
            &before_training_weights,
            &model.weights,
            model.config.hidden_dims,
            model.config.spatial_dims,
        );
        let train_motion_memory_output_delta_norm =
            growth_3d_velocity_channels(model.config.state_dims)
                .map(|channels| {
                    channels
                        .map(|channel| {
                            output_channel_delta_norm(
                                &before_training_weights,
                                &model.weights,
                                model.config.hidden_dims,
                                model.config.spatial_dims + channel,
                            )
                            .powi(2)
                        })
                        .sum::<f32>()
                        .sqrt()
                })
                .unwrap_or(0.0);
        let train_material_output_delta_norm =
            growth_3d_material_opacity_channel(model.config.state_dims)
                .map(|channel| {
                    output_channel_delta_norm(
                        &before_training_weights,
                        &model.weights,
                        model.config.hidden_dims,
                        model.config.spatial_dims + channel,
                    )
                })
                .unwrap_or(0.0);
        let train_color_output_delta_norm = growth_3d_color_output_channels(&model.config)
            .map(|channels| {
                channels
                    .into_iter()
                    .map(|channel| {
                        output_channel_delta_norm(
                            &before_training_weights,
                            &model.weights,
                            model.config.hidden_dims,
                            channel,
                        )
                        .powi(2)
                    })
                    .sum::<f32>()
                    .sqrt()
            })
            .unwrap_or(0.0);
        let after_trace = render_training_trace(model, grid, &cfg, round)?;
        let after = mesh_multiview_render_loss_from_trace(&after_trace, target, render_cfg)?;
        let selection = render_selection_metrics(
            model,
            grid,
            target,
            &cfg,
            render_cfg,
            Some(&selection_baseline),
        )?;
        let selected_checkpoint =
            render_selection_candidate_metrics_beats(&selection, &best_selection);
        if selected_checkpoint {
            best_model = model.clone();
            best_selection = selection.clone();
            selected_round = Some(round);
        }
        let continue_training_checkpoint = selected_checkpoint
            || render_selection_training_progress_beats(&selection, &before_selection);
        let rolled_back_to_best_checkpoint = !continue_training_checkpoint;
        let reported_train_step_scale = if rolled_back_to_best_checkpoint {
            0.0
        } else {
            train_step_scale
        };
        history.push(RenderProxyTrainingHistoryEntry {
            round,
            before_loss: before.total_loss,
            after_loss: after.total_loss,
            before_selection_loss: before_selection.render_loss,
            before_selection_score: before_selection.score,
            before_selection_density_psnr_db: before_selection.density_psnr_db,
            before_selection_min_active_extent_bbox_ratio: before_selection
                .min_active_extent_bbox_ratio,
            before_selection_min_active_extent_min_axis_ratio: before_selection
                .min_active_extent_min_axis_ratio,
            selection_loss: selection.render_loss,
            selection_score: selection.score,
            before_density_psnr_db: before.density_psnr_db,
            after_density_psnr_db: after.density_psnr_db,
            selection_density_psnr_db: selection.density_psnr_db,
            selection_active_surface_max: selection.active_surface_max,
            selection_target_coverage_fraction: selection.target_coverage_fraction,
            selection_material_visible_target_mean_distance: selection
                .material_visible_target_mean_distance,
            selection_material_visible_target_max_distance: selection
                .material_visible_target_max_distance,
            selection_material_visible_target_coverage_fraction: selection
                .material_visible_target_coverage_fraction,
            selection_strict_surface_active_count: selection.strict_surface_active_count,
            selection_strict_surface_materialized_fraction: selection
                .strict_surface_materialized_fraction,
            selection_strict_surface_material_mean_opacity: selection
                .strict_surface_material_mean_opacity,
            selection_strict_surface_material_visible_margin: selection
                .strict_surface_material_visible_margin,
            selection_strict_surface_material_max_visible_margin: selection
                .strict_surface_material_max_visible_margin,
            selection_material_visible_inactive_fraction: selection
                .material_visible_inactive_fraction,
            selection_material_visible_max_inactive_opacity: selection
                .material_visible_max_inactive_opacity,
            selection_material_active_mean_opacity: selection.material_active_mean_opacity,
            selection_material_visible_count: selection.material_visible_count,
            selection_active_color_state_mean_abs: selection.active_color_state_mean_abs,
            selection_active_color_state_max_abs: selection.active_color_state_max_abs,
            selection_active_color_state_stddev_mean: selection.active_color_state_stddev_mean,
            selection_surface_covered_bin_fraction: selection.surface_covered_bin_fraction,
            selection_surface_mean_bin_covered_fraction: selection
                .surface_mean_bin_covered_fraction,
            selection_material_visible_surface_covered_bin_fraction: selection
                .material_visible_surface_covered_bin_fraction,
            selection_material_visible_surface_mean_bin_covered_fraction: selection
                .material_visible_surface_mean_bin_covered_fraction,
            selection_surface_normal_covered_bin_fraction: selection
                .surface_normal_covered_bin_fraction,
            selection_surface_normal_mean_bin_covered_fraction: selection
                .surface_normal_mean_bin_covered_fraction,
            selection_material_visible_surface_normal_covered_bin_fraction: selection
                .material_visible_surface_normal_covered_bin_fraction,
            selection_material_visible_surface_normal_mean_bin_covered_fraction: selection
                .material_visible_surface_normal_mean_bin_covered_fraction,
            selection_material_visible_surface_tail_p99_distance: selection
                .material_visible_surface_tail_p99_distance,
            selection_material_visible_surface_tail_over_threshold_fraction: selection
                .material_visible_surface_tail_over_threshold_fraction,
            selection_min_active_extent_bbox_ratio: selection.min_active_extent_bbox_ratio,
            selection_min_active_extent_min_axis_ratio: selection.min_active_extent_min_axis_ratio,
            selection_min_final_active_count: selection.min_final_active_count,
            selection_min_newly_activated_fraction: selection.min_newly_activated_fraction,
            selection_min_front_local_newly_activated_fraction: selection
                .min_front_local_newly_activated_fraction,
            selection_max_front_liveness_margin: selection.max_front_liveness_margin,
            selection_min_front_liveness_candidate_count: selection
                .min_front_liveness_candidate_count,
            selection_max_extent_front_liveness_margin: selection.max_extent_front_liveness_margin,
            selection_min_extent_front_liveness_candidate_count: selection
                .min_extent_front_liveness_candidate_count,
            selection_max_temporal_front_liveness_margin: selection
                .max_temporal_front_liveness_margin,
            selection_min_temporal_front_liveness_candidate_count: selection
                .min_temporal_front_liveness_candidate_count,
            selection_max_temporal_extent_front_liveness_margin: selection
                .max_temporal_extent_front_liveness_margin,
            selection_min_temporal_extent_front_liveness_candidate_count: selection
                .min_temporal_extent_front_liveness_candidate_count,
            selection_max_temporal_activation_schedule_error: selection
                .max_temporal_activation_schedule_error,
            selection_all_temporal_activation_progressive: selection
                .all_temporal_activation_progressive,
            selection_all_temporal_geometry_progressive: selection
                .all_temporal_geometry_progressive,
            selection_morphology_non_regressed: selection.morphology_non_regressed,
            selected_checkpoint,
            rolled_back_to_best_checkpoint,
            selection_worst_seed: selection.worst_seed,
            selection_worst_failure_reasons: selection.worst_failure_reasons,
            before_color_psnr_db: before.color_psnr_db,
            after_color_psnr_db: after.color_psnr_db,
            before_depth_psnr_db: before.depth_psnr_db,
            after_depth_psnr_db: after.depth_psnr_db,
            train_initial_loss: train_report.initial_loss,
            train_final_loss: train_report.final_loss,
            train_best_loss: train_report.best_loss,
            supervised_loss: train_report.final_loss,
            train_step_count: train_report.steps,
            train_loss_history: train_report
                .history
                .iter()
                .map(|entry| entry.loss)
                .collect(),
            train_grad_norm: train_report
                .history
                .last()
                .map(|entry| entry.grad_norm)
                .unwrap_or(0.0),
            train_grad_norm_history: train_report
                .history
                .iter()
                .map(|entry| entry.grad_norm)
                .collect(),
            train_grad_scale: train_report
                .history
                .last()
                .map(|entry| entry.grad_scale)
                .unwrap_or(1.0),
            train_grad_scale_history: train_report
                .history
                .iter()
                .map(|entry| entry.grad_scale)
                .collect(),
            train_step_scale: reported_train_step_scale,
            direct_line_search_candidates,
            train_motion_output_delta_norm,
            train_motion_memory_output_delta_norm,
            train_liveness_output_delta_norm,
            train_phase_output_delta_norm,
            train_material_output_delta_norm,
            train_color_output_delta_norm,
            direct_objective_diagnostics,
            gradient_rms,
            opacity_gradient_rms,
            scale_gradient_rms,
        });
        if rolled_back_to_best_checkpoint {
            *model = best_model.clone();
        }
    }
    let final_render_loss = mesh_multiview_render_loss_from_trace(
        &render_training_trace(model, grid, &cfg, 0)?,
        target,
        render_cfg,
    )?;
    let final_trace = render_training_trace(model, grid, &cfg, 0)?;
    let final_gaussian_volume = gaussian_volume_stats_for_trace(&final_trace, render_cfg);

    Ok(RenderProxyTrainingReport {
        rounds: cfg.rounds,
        supervised_steps_per_round: cfg.supervised_steps_per_round,
        objective: render_training_objective_config(&cfg, render_cfg),
        gradient_particles: cfg.gradient_particles,
        gradient_mode: cfg.gradient_mode,
        finite_diff_eps: cfg.finite_diff_eps,
        motion_gain: cfg.motion_gain,
        perception_position_gain: cfg.perception_position_gain,
        max_update_norm: cfg.max_update_norm,
        trajectory_supervision: cfg.trajectory_supervision,
        trajectory_render_gain: cfg.trajectory_render_gain,
        trajectory_mesh_gain: cfg.trajectory_mesh_gain,
        trajectory_render_samples: cfg.trajectory_render_samples,
        liveness_gain: cfg.liveness_gain,
        liveness_front_radius: cfg.liveness_front_radius,
        liveness_update_multiplier: cfg.liveness_update_multiplier,
        coverage_gain: cfg.coverage_gain,
        coverage_samples: cfg.coverage_samples,
        coverage_mode: cfg.coverage_mode,
        coverage_softness: cfg.coverage_softness,
        coverage_repulsion_gain: cfg.coverage_repulsion_gain,
        coverage_gap_gain: cfg.coverage_gap_gain,
        coverage_repulsion_radius: cfg.coverage_repulsion_radius,
        coverage_normal_weight: cfg.coverage_normal_weight,
        extent_gain: cfg.extent_gain,
        full_coverage_adjoint: cfg.full_coverage_adjoint,
        surface_gain: cfg.surface_gain,
        surface_escape_gain: cfg.surface_escape_gain,
        opacity_gain: cfg.opacity_gain,
        material_liveness_gain: cfg.material_liveness_gain,
        material_tail_gain: cfg.material_tail_gain,
        material_suppression_update_multiplier: cfg.material_suppression_update_multiplier,
        material_max_opacity_update: cfg.material_max_opacity_update,
        scale_gain: cfg.scale_gain,
        scale_budget_weight: cfg.scale_budget_weight,
        max_opacity_update: cfg.max_opacity_update,
        direct_output_gradient_rms_cap: cfg.direct_output_gradient_rms_cap,
        direct_line_search: cfg.direct_line_search,
        direct_line_search_scales: sanitized_direct_line_search_scales(&cfg),
        direct_material_output_only: cfg.direct_material_output_only,
        training_backend: cfg.training_backend,
        direct_selection_seed_training: cfg.direct_selection_seed_training,
        selection_seed: cfg.selection_seed,
        selection_seeds: render_proxy_selection_seeds(&cfg),
        initial_gaussian_volume,
        final_gaussian_volume,
        initial_render_loss,
        final_render_loss,
        selected_round,
        history,
    })
}
