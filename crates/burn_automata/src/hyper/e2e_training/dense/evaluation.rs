//! Loss, geometry, PSNR, and adapter diagnostics used during validation.

use super::*;

    pub(super) fn evaluate_targets(
        params: &BurnBaseParams,
        adapters: &[BurnAdapterParams],
        targets: &[BurnTargetExample],
        config: DirectBasisTrainConfig,
        requested_examples: usize,
        seed: u64,
    ) -> Result<Option<CliHyper2dDirectBasisLossSummary>, Box<dyn std::error::Error>> {
        if targets.is_empty() {
            return Ok(None);
        }
        let mut indices = (0..targets.len()).collect::<Vec<_>>();
        if requested_examples > 0 && requested_examples < indices.len() {
            let mut rng = StdRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
            indices.truncate(requested_examples);
            indices.sort_unstable();
        }
        let mut summary = CliHyper2dDirectBasisLossSummary {
            examples: indices.len(),
            mean_total_loss: 0.0,
            max_total_loss: 0.0,
            mean_splat_loss: 0.0,
            mean_color_loss: 0.0,
            mean_density_loss: 0.0,
        };
        let eval_batch_size = normalized_eval_batch_size(config.eval_batch_size, indices.len());
        for chunk in indices.chunks(eval_batch_size) {
            if homogeneous_particle_count(targets, chunk).is_some() {
                let loss = batch_example_eval_loss(params, adapters, targets, chunk, config, seed)?;
                for scalars in loss_vector_scalars(loss)? {
                    summary.mean_total_loss += scalars.total;
                    summary.max_total_loss = summary.max_total_loss.max(scalars.total);
                    summary.mean_splat_loss += scalars.splat;
                    summary.mean_color_loss += scalars.color;
                    summary.mean_density_loss += scalars.density;
                }
            } else {
                for &idx in chunk {
                    let loss = example_eval_loss_bounded(
                        params,
                        &adapters[idx],
                        &targets[idx],
                        config,
                        seed.wrapping_add(idx as u64),
                    );
                    let scalars = loss_scalars(&loss)?;
                    summary.mean_total_loss += scalars.total;
                    summary.max_total_loss = summary.max_total_loss.max(scalars.total);
                    summary.mean_splat_loss += scalars.splat;
                    summary.mean_color_loss += scalars.color;
                    summary.mean_density_loss += scalars.density;
                }
            }
        }
        let scale = 1.0 / indices.len() as f32;
        summary.mean_total_loss *= scale;
        summary.mean_splat_loss *= scale;
        summary.mean_color_loss *= scale;
        summary.mean_density_loss *= scale;
        Ok(Some(summary))
    }

    pub(super) fn evaluate_target_geometry(
        params: &BurnBaseParams,
        adapters: &[BurnAdapterParams],
        targets: &[BurnTargetExample],
        config: DirectBasisTrainConfig,
        requested_examples: usize,
        seed: u64,
    ) -> Result<Option<BurnGeometrySummary>, Box<dyn std::error::Error>> {
        if targets.is_empty() {
            return Ok(None);
        }
        let mut indices = (0..targets.len()).collect::<Vec<_>>();
        if requested_examples > 0 && requested_examples < indices.len() {
            let mut rng = StdRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
            indices.truncate(requested_examples);
            indices.sort_unstable();
        }
        let eval_batch_size = normalized_eval_batch_size(config.eval_batch_size, indices.len());
        let mut total = BurnGeometrySummary {
            examples: 0,
            mean_score: 0.0,
            mean_foreground_iou: 0.0,
            mean_target_recall: 0.0,
            mean_generated_precision: 0.0,
            mean_bbox_iou: 0.0,
            mean_lit_pixel_ratio: 0.0,
            mean_bbox_width_ratio: 0.0,
            mean_bbox_area_ratio: 0.0,
        };
        for chunk in indices.chunks(eval_batch_size) {
            if homogeneous_particle_count(targets, chunk).is_none() {
                continue;
            }
            let Some(summary) =
                batch_example_geometry(params, adapters, targets, chunk, config, seed)?
            else {
                continue;
            };
            let weight = summary.examples as f32;
            total.examples += summary.examples;
            total.mean_score += summary.mean_score * weight;
            total.mean_foreground_iou += summary.mean_foreground_iou * weight;
            total.mean_target_recall += summary.mean_target_recall * weight;
            total.mean_generated_precision += summary.mean_generated_precision * weight;
            total.mean_bbox_iou += summary.mean_bbox_iou * weight;
            total.mean_lit_pixel_ratio += summary.mean_lit_pixel_ratio * weight;
            total.mean_bbox_width_ratio += summary.mean_bbox_width_ratio * weight;
            total.mean_bbox_area_ratio += summary.mean_bbox_area_ratio * weight;
        }
        if total.examples == 0 {
            return Ok(None);
        }
        let scale = 1.0 / total.examples as f32;
        total.mean_score *= scale;
        total.mean_foreground_iou *= scale;
        total.mean_target_recall *= scale;
        total.mean_generated_precision *= scale;
        total.mean_bbox_iou *= scale;
        total.mean_lit_pixel_ratio *= scale;
        total.mean_bbox_width_ratio *= scale;
        total.mean_bbox_area_ratio *= scale;
        Ok(Some(total))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_e2e_rollout_quality(
        params: &BurnBaseParams,
        generator: &BurnE2eGeneratorParams,
        npa_config: &NpaConfig,
        train_examples: &[BurnE2eRolloutExample],
        holdout_examples: &[BurnE2eRolloutExample],
        train_conditions: &BurnE2eConditionCache,
        holdout_conditions: &BurnE2eConditionCache,
        config: BurnE2eRolloutTrainConfig,
        device: &BurnDevice,
    ) -> Result<Option<BurnE2eRolloutQualityReport>, Box<dyn std::error::Error>> {
        let mut horizons = config.validation_horizons
            [..config.validation_horizon_count.min(config.validation_horizons.len())]
            .iter()
            .copied()
            .filter(|&steps| steps > 0)
            .collect::<Vec<_>>();
        horizons.push(config.validation_steps.max(1));
        horizons.sort_unstable();
        horizons.dedup();

        let mut final_report = None::<BurnE2eRolloutQualityReport>;
        let mut summaries = Vec::with_capacity(horizons.len());
        let mut total_elapsed_ms = 0.0_f64;
        let mut total_particle_steps = 0.0_f64;
        let mut total_adapter_batches = 0usize;
        for &steps in &horizons {
            let horizon_config = BurnE2eRolloutTrainConfig {
                validation_steps: steps,
                validation_horizon_count: 0,
                ..config
            };
            let Some(report) = evaluate_e2e_rollout_quality_single(
                params,
                generator,
                npa_config,
                train_examples,
                holdout_examples,
                train_conditions,
                holdout_conditions,
                horizon_config,
                device,
            )? else {
                return Ok(None);
            };
            total_elapsed_ms += report.elapsed_ms;
            total_particle_steps += report.particle_steps;
            total_adapter_batches = total_adapter_batches.saturating_add(report.adapter_batches);
            summaries.push(BurnE2eRolloutHorizonSummary {
                rollout_steps: steps,
                aggregate_composited_rgb_psnr_db: report.aggregate_composited_rgb_psnr_db,
                p10_composited_rgb_psnr_db: report.p10_composited_rgb_psnr_db,
                min_composited_rgb_psnr_db: report.min_composited_rgb_psnr_db,
                target_point_splat_p10_composited_rgb_psnr_db: report
                    .target_point_splat_p10_composited_rgb_psnr_db,
                p10_gap_to_target_point_splat_db: report.p10_gap_to_target_point_splat_db,
                aggregate_density_psnr_db: report.aggregate_density_psnr_db,
                mean_density_soft_iou: report.mean_density_soft_iou,
                condition_shuffle_composited_psnr_gap_db: report
                    .condition_shuffle_composited_psnr_gap_db,
                generated_adapter_composited_psnr_gain_db: report
                    .generated_adapter_composited_psnr_gain_db,
                mean_passed: report.mean_passed,
                all_examples_passed: report.all_examples_passed,
                conditional_control_passed: report.conditional_control_passed,
                passed: report.passed,
            });
            if steps == config.validation_steps.max(1) {
                final_report = Some(report);
            }
        }

        let mut report = final_report.expect("final validation horizon is always evaluated");
        let selection_summaries = summaries
            .iter()
            .filter(|summary| {
                summary.rollout_steps >= config.validation_selection_horizon_min_steps
            })
            .collect::<Vec<_>>();
        debug_assert!(!selection_summaries.is_empty());
        let selection_psnr_db = selection_summaries
            .iter()
            .map(|summary| summary.p10_composited_rgb_psnr_db)
            .fold(f32::INFINITY, f32::min);
        let peak_horizon_p10_composited_rgb_psnr_db = summaries
            .iter()
            .map(|summary| summary.p10_composited_rgb_psnr_db)
            .fold(f32::NEG_INFINITY, f32::max);
        let final_horizon_p10_composited_rgb_psnr_db = report.p10_composited_rgb_psnr_db;
        report.selection_metric = "min-horizon-p10-composited-rgb-psnr";
        report.selection_psnr_db = selection_psnr_db;
        report.selection_horizon_min_steps = config.validation_selection_horizon_min_steps;
        report.passed = selection_summaries.iter().all(|summary| summary.passed)
            && selection_psnr_db >= config.validation_psnr_threshold_db;
        report.mean_passed = selection_summaries
            .iter()
            .all(|summary| summary.mean_passed);
        report.all_examples_passed = selection_summaries
            .iter()
            .all(|summary| summary.all_examples_passed);
        report.conditional_control_passed = selection_summaries
            .iter()
            .all(|summary| summary.conditional_control_passed);
        report.elapsed_ms = total_elapsed_ms;
        report.particle_steps = total_particle_steps;
        report.particle_steps_per_sec = total_particle_steps
            / (total_elapsed_ms / 1000.0).max(f64::MIN_POSITIVE);
        report.dense_pair_interactions_per_sec =
            report.particle_steps_per_sec * config.validation_particles as f64;
        report.adapter_batches = total_adapter_batches;
        report.horizon_summaries = summaries;
        report.peak_horizon_p10_composited_rgb_psnr_db =
            peak_horizon_p10_composited_rgb_psnr_db;
        report.final_horizon_p10_composited_rgb_psnr_db =
            final_horizon_p10_composited_rgb_psnr_db;
        report.peak_to_final_p10_drop_db =
            peak_horizon_p10_composited_rgb_psnr_db - final_horizon_p10_composited_rgb_psnr_db;
        Ok(Some(report))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_e2e_rollout_quality_single(
        params: &BurnBaseParams,
        generator: &BurnE2eGeneratorParams,
        npa_config: &NpaConfig,
        train_examples: &[BurnE2eRolloutExample],
        holdout_examples: &[BurnE2eRolloutExample],
        train_conditions: &BurnE2eConditionCache,
        holdout_conditions: &BurnE2eConditionCache,
        config: BurnE2eRolloutTrainConfig,
        device: &BurnDevice,
    ) -> Result<Option<BurnE2eRolloutQualityReport>, Box<dyn std::error::Error>> {
        if config.validation_examples == 0 {
            return Ok(None);
        }
        let (split, examples, conditions) = if holdout_examples.is_empty() {
            ("train", train_examples, train_conditions)
        } else {
            ("holdout", holdout_examples, holdout_conditions)
        };
        if examples.is_empty() {
            return Ok(None);
        }
        let started = Instant::now();
        let mut indices = (0..examples.len()).collect::<Vec<_>>();
        if config.validation_examples < indices.len() {
            let mut rng = StdRng::seed_from_u64(config.validation_seed);
            indices.shuffle(&mut rng);
            indices.truncate(config.validation_examples);
            indices.sort_unstable();
        }
        let eval_config = validation_direct_config(config);
        let targets = burn_e2e_targets_for_indices_with_runtime(
            examples,
            &indices,
            config,
            device,
            Some(config.validation_particles),
            Some(config.validation_update_prob),
        )?;
        let target_indices = (0..targets.len()).collect::<Vec<_>>();
        let eval_batch_size = normalized_eval_batch_size(eval_config.eval_batch_size, indices.len());
        let mut entries = Vec::with_capacity(indices.len());
        let dense_row_residual = generator.row_flow.is_some();
        let adapter_parameter_count = if dense_row_residual {
            NpaParameterRowLayout2d::new(npa_config).parameter_count()
        } else {
            NpaLowRankAdapter::parameter_count_for_config(npa_config, config.adapter_rank)
        };
        let mut adapter_parameter_rows = Vec::with_capacity(indices.len());
        let mut adapter_batches = 0usize;
        for (condition_chunk, target_chunk) in indices
            .chunks(eval_batch_size)
            .zip(target_indices.chunks(eval_batch_size))
        {
            adapter_batches += 1;
            let quality = batch_e2e_eval_quality(
                params,
                generator,
                npa_config,
                conditions,
                &targets,
                condition_chunk,
                target_chunk,
                config,
                eval_config,
                config.validation_seed,
                device,
                E2eEvalConditionMode::Generated,
            )?;
            let adapter_values = tensor_vec(
                quality
                    .adapter_vector
                    .clone()
                    .expect("generated quality batch contains adapter vectors")
                    .inner(),
            )?;
            if adapter_values.len() != condition_chunk.len() * adapter_parameter_count {
                return Err(std::io::Error::other(
                    "HyperNPA e2e adapter diagnostics readback length mismatch",
                )
                .into());
            }
            adapter_parameter_rows.extend(
                adapter_values
                    .chunks_exact(adapter_parameter_count)
                    .map(<[f32]>::to_vec),
            );
            let losses = loss_vector_scalars(quality.loss)?;
            let render_rgb_mses = tensor1_vec(quality.render_rgb_mse.inner())?;
            let composited_rgb_mses = tensor1_vec(quality.composited_rgb_mse.inner())?;
            let foreground_rgb_mses = tensor1_vec(quality.foreground_rgb_mse.inner())?;
            let density_mses = tensor1_vec(quality.density_mse.inner())?;
            let density_soft_ious = tensor1_vec(quality.density_soft_iou.inner())?;
            if [
                losses.len(),
                render_rgb_mses.len(),
                composited_rgb_mses.len(),
                foreground_rgb_mses.len(),
                density_mses.len(),
                density_soft_ious.len(),
            ]
            .into_iter()
            .any(|len| len != condition_chunk.len())
            {
                return Err(std::io::Error::other(
                    "HyperNPA e2e quality readback length mismatch",
                )
                .into());
            }
            for local in 0..condition_chunk.len() {
                let idx = condition_chunk[local];
                let loss = losses[local];
                let render_rgb_mse =
                    finite_scalar("HyperNPA e2e render RGB MSE", render_rgb_mses[local])?;
                let render_rgb_psnr_db = psnr_db_from_mse(render_rgb_mse);
                let composited_rgb_mse = finite_scalar(
                    "HyperNPA e2e composited RGB MSE",
                    composited_rgb_mses[local],
                )?;
                let composited_rgb_psnr_db = psnr_db_from_mse(composited_rgb_mse);
                let foreground_rgb_mse = finite_scalar(
                    "HyperNPA e2e foreground RGB MSE",
                    foreground_rgb_mses[local],
                )?;
                let foreground_rgb_psnr_db = psnr_db_from_mse(foreground_rgb_mse);
                let density_mse =
                    finite_scalar("HyperNPA e2e density MSE", density_mses[local])?;
                let density_psnr_db = psnr_db_from_mse(density_mse);
                let density_soft_iou = finite_scalar(
                    "HyperNPA e2e density soft IoU",
                    density_soft_ious[local],
                )?;
                entries.push(BurnE2eRolloutQualityEntry {
                    slug: examples[idx].slug.clone(),
                    total_loss: loss.total,
                    splat_loss: loss.splat,
                    color_loss: loss.color,
                    density_loss: loss.density,
                    render_rgb_mse,
                    render_rgb_psnr_db,
                    composited_rgb_mse,
                    composited_rgb_psnr_db,
                    foreground_rgb_mse,
                    foreground_rgb_psnr_db,
                    density_mse,
                    density_psnr_db,
                    density_soft_iou,
                    passed: composited_rgb_psnr_db >= config.validation_psnr_threshold_db,
                });
            }
        }
        let (mean_condition_shuffle_render_rgb_psnr_db, condition_shuffle_composited_rgb_psnr_db) = if indices.len() > 1 {
            let mut shuffled_condition_indices = indices.clone();
            shuffled_condition_indices.rotate_left(1);
            let mut psnr_sum = 0.0_f32;
            let mut composited_mse_sum = 0.0_f32;
            let mut psnr_count = 0usize;
            for (condition_chunk, target_chunk) in shuffled_condition_indices
                .chunks(eval_batch_size)
                .zip(target_indices.chunks(eval_batch_size))
            {
                let quality = batch_e2e_eval_quality(
                    params,
                    generator,
                    npa_config,
                    conditions,
                    &targets,
                    condition_chunk,
                    target_chunk,
                    config,
                    eval_config,
                    config.validation_seed,
                    device,
                    E2eEvalConditionMode::Generated,
                )?;
                let render_rgb_mses = tensor1_vec(quality.render_rgb_mse.inner())?;
                let composited_rgb_mses = tensor1_vec(quality.composited_rgb_mse.inner())?;
                for (render_rgb_mse, composited_rgb_mse) in
                    render_rgb_mses.into_iter().zip(composited_rgb_mses)
                {
                    let render_rgb_mse =
                        finite_scalar("HyperNPA e2e shuffled-condition render RGB MSE", render_rgb_mse)?;
                    let composited_rgb_mse = finite_scalar(
                        "HyperNPA e2e shuffled-condition composited RGB MSE",
                        composited_rgb_mse,
                    )?;
                    psnr_sum += psnr_db_from_mse(render_rgb_mse);
                    composited_mse_sum += composited_rgb_mse;
                    psnr_count += 1;
                }
            }
            if psnr_count > 0 {
                (
                    Some(psnr_sum / psnr_count as f32),
                    Some(psnr_db_from_mse(composited_mse_sum / psnr_count as f32)),
                )
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        let mut base_only_composited_mse_sum = 0.0_f32;
        let mut base_only_density_mse_sum = 0.0_f32;
        let mut base_only_density_soft_iou_sum = 0.0_f32;
        let mut base_only_count = 0usize;
        for (condition_chunk, target_chunk) in indices
            .chunks(eval_batch_size)
            .zip(target_indices.chunks(eval_batch_size))
        {
            let quality = batch_e2e_eval_quality(
                params,
                generator,
                npa_config,
                conditions,
                &targets,
                condition_chunk,
                target_chunk,
                config,
                eval_config,
                config.validation_seed,
                device,
                E2eEvalConditionMode::BaseOnly,
            )?;
            let composited_mses = tensor1_vec(quality.composited_rgb_mse.inner())?;
            let density_mses = tensor1_vec(quality.density_mse.inner())?;
            let density_soft_ious = tensor1_vec(quality.density_soft_iou.inner())?;
            for local in 0..composited_mses.len() {
                base_only_composited_mse_sum += finite_scalar(
                    "HyperNPA e2e base-only composited RGB MSE",
                    composited_mses[local],
                )?;
                base_only_density_mse_sum += finite_scalar(
                    "HyperNPA e2e base-only density MSE",
                    density_mses[local],
                )?;
                base_only_density_soft_iou_sum += finite_scalar(
                    "HyperNPA e2e base-only density soft IoU",
                    density_soft_ious[local],
                )?;
                base_only_count += 1;
            }
        }
        let mut dino_nearest_teacher_entries = Vec::new();
        let (
            dino_nearest_teacher_render_rgb_psnr_db,
            dino_nearest_teacher_composited_rgb_psnr_db,
        ) = if split == "holdout" && train_conditions.teacher_vectors.is_some() {
            let nearest = train_conditions.nearest_rows(conditions, &indices)?;
            let mut render_psnr_sum = 0.0_f32;
            let mut composited_mse_sum = 0.0_f32;
            let mut count = 0usize;
            for start in (0..indices.len()).step_by(eval_batch_size) {
                let end = (start + eval_batch_size).min(indices.len());
                let condition_chunk = &indices[start..end];
                let target_chunk = &target_indices[start..end];
                let nearest_chunk = &nearest[start..end];
                let nearest_indices = nearest_chunk
                    .iter()
                    .map(|(idx, _)| *idx)
                    .collect::<Vec<_>>();
                let teacher_vectors = train_conditions
                    .select_teacher(&nearest_indices)
                    .expect("nearest-teacher validation requires train teacher adapters");
                let quality = batch_e2e_eval_quality(
                    params,
                    generator,
                    npa_config,
                    conditions,
                    &targets,
                    condition_chunk,
                    target_chunk,
                    config,
                    eval_config,
                    config.validation_seed,
                    device,
                    E2eEvalConditionMode::ExplicitAdapter(teacher_vectors),
                )?;
                let render_rgb_mses = tensor1_vec(quality.render_rgb_mse.inner())?;
                let composited_rgb_mses = tensor1_vec(quality.composited_rgb_mse.inner())?;
                for local in 0..condition_chunk.len() {
                    let render_rgb_mse = finite_scalar(
                        "HyperNPA e2e nearest-teacher render RGB MSE",
                        render_rgb_mses[local],
                    )?;
                    let composited_rgb_mse = finite_scalar(
                        "HyperNPA e2e nearest-teacher composited RGB MSE",
                        composited_rgb_mses[local],
                    )?;
                    let (train_idx, condition_l2_distance) = nearest_chunk[local];
                    let render_rgb_psnr_db = psnr_db_from_mse(render_rgb_mse);
                    let composited_rgb_psnr_db = psnr_db_from_mse(composited_rgb_mse);
                    render_psnr_sum += render_rgb_psnr_db;
                    composited_mse_sum += composited_rgb_mse;
                    count += 1;
                    dino_nearest_teacher_entries.push(BurnE2eNearestTeacherEntry {
                        holdout_slug: examples[condition_chunk[local]].slug.clone(),
                        train_slug: train_examples[train_idx].slug.clone(),
                        condition_l2_distance,
                        render_rgb_psnr_db,
                        composited_rgb_psnr_db,
                    });
                }
            }
            if count > 0 {
                (
                    Some(render_psnr_sum / count as f32),
                    Some(psnr_db_from_mse(composited_mse_sum / count as f32)),
                )
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        let examples_count = entries.len();
        if examples_count == 0 {
            return Ok(None);
        }
        let target_point_splat_mses = target_point_splat_composited_mses(
            &targets,
            eval_config,
            eval_batch_size,
            device,
        )?;
        let target_point_splat_aggregate_composited_rgb_psnr_db = psnr_db_from_mse(
            target_point_splat_mses.iter().sum::<f32>()
                / target_point_splat_mses.len().max(1) as f32,
        );
        let mut target_point_splat_psnrs = target_point_splat_mses
            .into_iter()
            .map(psnr_db_from_mse)
            .collect::<Vec<_>>();
        target_point_splat_psnrs.sort_by(f32::total_cmp);
        let target_point_splat_p10_composited_rgb_psnr_db =
            sorted_percentile(&target_point_splat_psnrs, 0.1);
        let mut mean_total_loss = 0.0_f32;
        let mut mean_splat_loss = 0.0_f32;
        let mut mean_color_loss = 0.0_f32;
        let mut mean_density_loss = 0.0_f32;
        let mut mean_render_rgb_mse = 0.0_f32;
        let mut mean_render_rgb_psnr_db = 0.0_f32;
        let mut min_render_rgb_psnr_db = f32::INFINITY;
        let mut max_render_rgb_psnr_db = f32::NEG_INFINITY;
        let mut aggregate_composited_rgb_mse = 0.0_f32;
        let mut aggregate_foreground_rgb_mse = 0.0_f32;
        let mut aggregate_density_mse = 0.0_f32;
        let mut mean_density_soft_iou = 0.0_f32;
        let mut composited_psnrs = Vec::with_capacity(examples_count);
        for entry in &entries {
            mean_total_loss += entry.total_loss;
            mean_splat_loss += entry.splat_loss;
            mean_color_loss += entry.color_loss;
            mean_density_loss += entry.density_loss;
            mean_render_rgb_mse += entry.render_rgb_mse;
            mean_render_rgb_psnr_db += entry.render_rgb_psnr_db;
            min_render_rgb_psnr_db = min_render_rgb_psnr_db.min(entry.render_rgb_psnr_db);
            max_render_rgb_psnr_db = max_render_rgb_psnr_db.max(entry.render_rgb_psnr_db);
            aggregate_composited_rgb_mse += entry.composited_rgb_mse;
            aggregate_foreground_rgb_mse += entry.foreground_rgb_mse;
            aggregate_density_mse += entry.density_mse;
            mean_density_soft_iou += entry.density_soft_iou;
            composited_psnrs.push(entry.composited_rgb_psnr_db);
        }
        let scale = 1.0 / examples_count as f32;
        mean_total_loss *= scale;
        mean_splat_loss *= scale;
        mean_color_loss *= scale;
        mean_density_loss *= scale;
        mean_render_rgb_mse *= scale;
        mean_render_rgb_psnr_db *= scale;
        aggregate_composited_rgb_mse *= scale;
        aggregate_foreground_rgb_mse *= scale;
        aggregate_density_mse *= scale;
        mean_density_soft_iou *= scale;
        composited_psnrs.sort_by(f32::total_cmp);
        let mean_composited_rgb_psnr_db = composited_psnrs.iter().sum::<f32>() * scale;
        let median_composited_rgb_psnr_db = sorted_percentile(&composited_psnrs, 0.5);
        let p10_composited_rgb_psnr_db = sorted_percentile(&composited_psnrs, 0.1);
        let min_composited_rgb_psnr_db = composited_psnrs[0];
        let max_composited_rgb_psnr_db = composited_psnrs[examples_count - 1];
        let aggregate_composited_rgb_psnr_db =
            psnr_db_from_mse(aggregate_composited_rgb_mse);
        let aggregate_foreground_rgb_psnr_db =
            psnr_db_from_mse(aggregate_foreground_rgb_mse);
        let aggregate_density_psnr_db = psnr_db_from_mse(aggregate_density_mse);
        let base_only_scale = 1.0 / base_only_count.max(1) as f32;
        let base_only_composited_rgb_psnr_db =
            psnr_db_from_mse(base_only_composited_mse_sum * base_only_scale);
        let base_only_density_psnr_db =
            psnr_db_from_mse(base_only_density_mse_sum * base_only_scale);
        let base_only_density_soft_iou = base_only_density_soft_iou_sum * base_only_scale;
        let generated_adapter_composited_psnr_gain_db =
            aggregate_composited_rgb_psnr_db - base_only_composited_rgb_psnr_db;
        let selection_psnr_db = p10_composited_rgb_psnr_db;
        let p10_gap_to_target_point_splat_db =
            target_point_splat_p10_composited_rgb_psnr_db - p10_composited_rgb_psnr_db;
        let condition_shuffle_psnr_gap_db =
            mean_condition_shuffle_render_rgb_psnr_db.map(|shuffle| mean_render_rgb_psnr_db - shuffle);
        let condition_shuffle_composited_psnr_gap_db = condition_shuffle_composited_rgb_psnr_db
            .map(|shuffle| aggregate_composited_rgb_psnr_db - shuffle);
        let mean_passed =
            aggregate_composited_rgb_psnr_db >= config.validation_psnr_threshold_db;
        let all_examples_passed = entries.iter().all(|entry| entry.passed);
        let conditional_control_passed = generated_adapter_composited_psnr_gain_db > 0.0
            && condition_shuffle_composited_psnr_gap_db.is_none_or(|gap| gap > 0.0);
        let adapter_diagnostics = adapter_diagnostics(
            &adapter_parameter_rows,
            npa_config,
            config.adapter_rank,
            dense_row_residual,
        )?;
        let passed = mean_passed
            && selection_psnr_db >= config.validation_psnr_threshold_db
            && conditional_control_passed;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let particle_steps =
            examples_count as f64 * config.validation_particles as f64 * config.validation_steps as f64;
        let particle_steps_per_sec =
            particle_steps / (elapsed_ms / 1000.0).max(f64::MIN_POSITIVE);
        let dense_pair_interactions_per_sec =
            particle_steps_per_sec * config.validation_particles as f64;
        Ok(Some(BurnE2eRolloutQualityReport {
            split,
            examples: examples_count,
            particle_count: config.validation_particles,
            rollout_steps: config.validation_steps,
            update_prob: config.validation_update_prob,
            seed: config.validation_seed,
            psnr_threshold_db: config.validation_psnr_threshold_db,
            passed,
            mean_passed,
            all_examples_passed,
            elapsed_ms,
            particle_steps,
            particle_steps_per_sec,
            dense_pair_interactions_per_sec,
            adapter_batches,
            mean_total_loss,
            mean_splat_loss,
            mean_color_loss,
            mean_density_loss,
            mean_render_rgb_mse,
            mean_render_rgb_psnr_db,
            min_render_rgb_psnr_db,
            max_render_rgb_psnr_db,
            selection_metric: "p10-composited-rgb-psnr",
            selection_psnr_db,
            selection_horizon_min_steps: config.validation_steps,
            horizon_summaries: Vec::new(),
            peak_horizon_p10_composited_rgb_psnr_db: p10_composited_rgb_psnr_db,
            final_horizon_p10_composited_rgb_psnr_db: p10_composited_rgb_psnr_db,
            peak_to_final_p10_drop_db: 0.0,
            target_point_splat_aggregate_composited_rgb_psnr_db,
            target_point_splat_p10_composited_rgb_psnr_db,
            p10_gap_to_target_point_splat_db,
            aggregate_composited_rgb_mse,
            aggregate_composited_rgb_psnr_db,
            mean_composited_rgb_psnr_db,
            median_composited_rgb_psnr_db,
            p10_composited_rgb_psnr_db,
            min_composited_rgb_psnr_db,
            max_composited_rgb_psnr_db,
            aggregate_foreground_rgb_mse,
            aggregate_foreground_rgb_psnr_db,
            aggregate_density_mse,
            aggregate_density_psnr_db,
            mean_density_soft_iou,
            mean_condition_shuffle_render_rgb_psnr_db,
            condition_shuffle_psnr_gap_db,
            condition_shuffle_composited_rgb_psnr_db,
            condition_shuffle_composited_psnr_gap_db,
            base_only_composited_rgb_psnr_db,
            generated_adapter_composited_psnr_gain_db,
            base_only_density_psnr_db,
            base_only_density_soft_iou,
            dino_nearest_teacher_render_rgb_psnr_db,
            dino_nearest_teacher_composited_rgb_psnr_db,
            dino_nearest_teacher_entries,
            conditional_control_passed,
            adapter_diagnostics,
            entries,
        }))
    }

    pub(super) struct BurnE2eQualityBatchTensors {
        pub(super) loss: BurnLossBatchTensors,
        pub(super) adapter_vector: Option<Tensor2>,
        pub(super) render_rgb_mse: Tensor1,
        pub(super) composited_rgb_mse: Tensor1,
        pub(super) foreground_rgb_mse: Tensor1,
        pub(super) density_mse: Tensor1,
        pub(super) density_soft_iou: Tensor1,
    }

    #[derive(Clone)]
    pub(super) enum E2eEvalConditionMode {
        Generated,
        BaseOnly,
        ExplicitAdapter(Tensor2),
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn batch_e2e_eval_quality(
        params: &BurnBaseParams,
        generator: &BurnE2eGeneratorParams,
        npa_config: &NpaConfig,
        conditions: &BurnE2eConditionCache,
        targets: &[BurnTargetExample],
        condition_indices: &[usize],
        target_indices: &[usize],
        generator_config: BurnE2eRolloutTrainConfig,
        eval_config: DirectBasisTrainConfig,
        seed: u64,
        device: &BurnDevice,
        condition_mode: E2eEvalConditionMode,
    ) -> Result<BurnE2eQualityBatchTensors, Box<dyn std::error::Error>> {
        if condition_indices.len() != target_indices.len() {
            return Err(std::io::Error::other(
                "HyperNPA e2e quality validation condition/target batch length mismatch",
            )
            .into());
        }
        let Some(particle_count) = homogeneous_particle_count(targets, target_indices) else {
            return Err(std::io::Error::other(
                "HyperNPA e2e quality validation requires homogeneous particle counts",
            )
            .into());
        };
        let condition = conditions.select(condition_indices)?;
        let (adapter_batch, condition_control, adapter_vector) = match condition_mode {
            E2eEvalConditionMode::Generated => {
                let adapter =
                    generator.adapter_batch(condition.clone(), npa_config, generator_config);
                let vector = if generator.row_flow.is_some() {
                    adapter.dense_residual_vector(npa_config)
                } else {
                    adapter.to_parameter_vector()
                };
                (
                    adapter,
                    generator.condition_control_batch(condition.clone(), generator_config),
                    Some(vector),
                )
            }
            E2eEvalConditionMode::BaseOnly => {
                let parameter_count = NpaLowRankAdapter::parameter_count_for_config(
                    npa_config,
                    generator_config.adapter_rank,
                );
                (
                    BurnAdapterBatch::from_parameter_vector(
                        Tensor::<BurnBackend, 2>::zeros(
                            [condition_indices.len(), parameter_count],
                            device,
                        ),
                        npa_config,
                        generator_config.adapter_rank,
                        generator_config.adapter_alpha,
                    ),
                    None,
                    None,
                )
            }
            E2eEvalConditionMode::ExplicitAdapter(vector) => (
                BurnAdapterBatch::from_parameter_vector(
                    vector,
                    npa_config,
                    generator_config.adapter_rank,
                    generator_config.adapter_alpha,
                ),
                None,
                None,
            ),
        };
        let (mut x, mut s) = seed_batch_tensors_with_seed_indices(
            targets,
            target_indices,
            target_indices,
            particle_count,
            eval_config,
            seed,
            device,
        );
        let mut rngs = target_indices
            .iter()
            .map(|idx| StdRng::seed_from_u64(seed.wrapping_add(*idx as u64) ^ 0x005e_ed2d))
            .collect::<Vec<_>>();
        let mut displacement = Tensor::<BurnBackend, 1>::zeros([target_indices.len()], device);
        let chunk_steps = tbptt_chunk_steps(eval_config);
        let mut remaining_steps = eval_config.rollout_steps;
        while remaining_steps > 0 {
            let steps = remaining_steps.min(chunk_steps);
            (x, s, displacement) = rollout_batch_eval_chunk(
                params,
                &adapter_batch,
                targets,
                target_indices,
                x,
                s,
                eval_config,
                particle_count,
                &mut rngs,
                steps,
                displacement,
                condition_control.as_ref(),
            );
            remaining_steps -= steps;
            if remaining_steps > 0 {
                x = detach3(x);
                s = detach3(s);
                displacement = detach1(displacement);
            }
        }
        let mut quality = target_splat_quality_batch_vector(
            &x,
            &s,
            targets,
            target_indices,
            eval_config,
            &adapter_batch,
            displacement,
        );
        quality.adapter_vector = adapter_vector;
        Ok(quality)
    }

    pub(super) fn normalized_eval_batch_size(requested: usize, examples: usize) -> usize {
        if requested == 0 {
            examples.max(1)
        } else {
            requested.min(examples).max(1)
        }
    }

    pub(super) fn e2e_lr_scale(config: BurnE2eRolloutTrainConfig, step: usize) -> f32 {
        let min_scale = config.min_lr_scale.clamp(0.0, 1.0);
        let schedule_scale = if config.steps <= 1 {
            1.0
        } else {
            let progress =
                step.saturating_sub(1) as f32 / config.steps.saturating_sub(1) as f32;
            let raw_scale = match config.lr_schedule {
                E2eLrSchedule::Constant => 1.0,
                E2eLrSchedule::Linear => 1.0 - progress,
                E2eLrSchedule::Cosine => {
                    0.5 * (1.0 + (std::f32::consts::PI * progress).cos())
                }
                E2eLrSchedule::UpstreamGrowing => {
                    let phase_step = step.saturating_sub(1) % 10_000 + 1;
                    let milestones_passed =
                        phase_step.saturating_sub(1).div_euclid(2_000).min(4);
                    0.3_f32.powi(milestones_passed as i32)
                }
            };
            min_scale + (1.0 - min_scale) * raw_scale.clamp(0.0, 1.0)
        };
        schedule_scale * e2e_lr_warmup_scale(config.lr_warmup_steps, step)
    }

    pub(super) fn e2e_lr_warmup_scale(warmup_steps: usize, step: usize) -> f32 {
        if warmup_steps == 0 {
            1.0
        } else {
            (step.max(1) as f32 / warmup_steps as f32).clamp(0.0, 1.0)
        }
    }

    pub(super) fn e2e_config_with_lr_scale(
        mut config: BurnE2eRolloutTrainConfig,
        lr_scale: f32,
    ) -> BurnE2eRolloutTrainConfig {
        let lr_scale = lr_scale.clamp(0.0, 1.0);
        config.base_optimizer.learning_rate *= lr_scale;
        config.generator_optimizer.learning_rate *= lr_scale;
        config
    }

    pub(super) fn reported_particle_step_speed_summary(
        history: &[BurnE2eRolloutHistoryEntry],
    ) -> (f64, f64, f64) {
        let mut speeds = history
            .iter()
            .map(|entry| entry.particle_steps_per_sec)
            .filter(|speed| speed.is_finite())
            .collect::<Vec<_>>();
        speeds.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
        let min = speeds.first().copied().unwrap_or_default();
        let median = speeds.get(speeds.len() / 2).copied().unwrap_or_default();
        let max = speeds.last().copied().unwrap_or_default();
        (min, median, max)
    }

    pub(super) fn reported_loss_summary(
        history: &[BurnE2eRolloutHistoryEntry],
    ) -> (Option<f32>, Option<f32>, usize, Option<f32>) {
        let first = history
            .iter()
            .find(|entry| entry.loss.is_finite())
            .map(|entry| entry.loss);
        let final_loss = history
            .iter()
            .rev()
            .find(|entry| entry.loss.is_finite())
            .map(|entry| entry.loss);
        let Some(best) = history
            .iter()
            .filter(|entry| entry.loss.is_finite())
            .min_by(|lhs, rhs| lhs.loss.total_cmp(&rhs.loss))
        else {
            return (first, None, 0, final_loss);
        };
        (first, Some(best.loss), best.step, final_loss)
    }

    pub(super) fn format_optional_f32(value: Option<f32>) -> String {
        value.map_or_else(|| "n/a".to_string(), |value| format!("{value:.3}"))
    }

    pub(super) fn psnr_db_from_mse(mse: f32) -> f32 {
        let mse = mse.max(1.0e-12);
        finite_scalar("HyperNPA e2e PSNR", 10.0 * (1.0 / mse).log10()).unwrap_or(0.0)
    }

    pub(super) fn sorted_percentile(sorted: &[f32], quantile: f32) -> f32 {
        debug_assert!(!sorted.is_empty());
        let position = quantile.clamp(0.0, 1.0) * sorted.len().saturating_sub(1) as f32;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;
        let blend = position - lower as f32;
        sorted[lower] * (1.0 - blend) + sorted[upper] * blend
    }

    pub(super) fn adapter_diagnostics(
        rows: &[Vec<f32>],
        npa_config: &NpaConfig,
        rank: usize,
        dense_row_residual: bool,
    ) -> AutomataResult<BurnE2eAdapterDiagnostics> {
        let transport_parameter_count =
            NpaLowRankAdapter::parameter_count_for_config(npa_config, rank);
        let dense_controller_dims = dense_row_residual
            .then(|| NpaParameterRowLayout2d::new(npa_config).parameter_count());
        let parameter_count = dense_controller_dims.unwrap_or(transport_parameter_count);
        if rows.is_empty()
            || rows.iter().any(|row| {
                row.len() != parameter_count || !row.iter().all(|value| value.is_finite())
            })
        {
            return Err(AutomataError::InvalidArgument(
                "adapter diagnostics require non-empty, finite, shape-consistent rows".to_string(),
            ));
        }
        let norms = rows
            .iter()
            .map(|row| row.iter().map(|value| value * value).sum::<f32>().sqrt())
            .collect::<Vec<_>>();
        let mean_l2_norm = norms.iter().sum::<f32>() / norms.len() as f32;
        let min_l2_norm = norms.iter().copied().fold(f32::INFINITY, f32::min);
        let max_l2_norm = norms.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut pairwise_distance_sum = 0.0_f32;
        let mut pairwise_cosine_sum = 0.0_f32;
        let mut min_pairwise_l2_distance = f32::INFINITY;
        let mut pairs = 0usize;
        for left in 0..rows.len() {
            for right in left + 1..rows.len() {
                let mut squared_distance = 0.0_f32;
                let mut dot = 0.0_f32;
                for (&lhs, &rhs) in rows[left].iter().zip(&rows[right]) {
                    let delta = lhs - rhs;
                    squared_distance += delta * delta;
                    dot += lhs * rhs;
                }
                let distance = squared_distance.sqrt();
                pairwise_distance_sum += distance;
                min_pairwise_l2_distance = min_pairwise_l2_distance.min(distance);
                pairwise_cosine_sum += dot / (norms[left] * norms[right]).max(EPSILON);
                pairs += 1;
            }
        }
        let range_rms = |offset: usize, len: usize| {
            let sum = rows
                .iter()
                .flat_map(|row| row[offset..offset + len].iter().copied())
                .map(|value| value * value)
                .sum::<f32>();
            (sum / (rows.len() * len).max(1) as f32).sqrt()
        };
        let (w1_dense_rms, w2_dense_rms, factor_rms, b1_rms, b2_rms) =
            if dense_row_residual {
                let p = npa_config.perception_dims();
                let h = npa_config.hidden_dims;
                let u = npa_config.update_dims();
                let w1_len = h * p;
                let b1_offset = w1_len;
                let w2_offset = b1_offset + h;
                let w2_len = u * h;
                let b2_offset = w2_offset + w2_len;
                (
                    Some(range_rms(0, w1_len)),
                    Some(range_rms(w2_offset, w2_len)),
                    None,
                    range_rms(b1_offset, h),
                    range_rms(b2_offset, u),
                )
            } else {
                let layout = crate::hyper::adapter_layout::AdapterParameterLayout2d::new(
                    npa_config, rank, 1,
                )?;
                let group_rms = |group| {
                    let segment = layout
                        .segments
                        .iter()
                        .find(|segment| segment.group == group)
                        .expect("adapter layout contains every parameter group");
                    range_rms(segment.vector_offset, segment.len)
                };
                use crate::hyper::adapter_layout::AdapterParameterGroup2d;
                (
                    None,
                    None,
                    Some([
                        group_rms(AdapterParameterGroup2d::W1Down),
                        group_rms(AdapterParameterGroup2d::W1Up),
                        group_rms(AdapterParameterGroup2d::W2Down),
                        group_rms(AdapterParameterGroup2d::W2Up),
                    ]),
                    group_rms(AdapterParameterGroup2d::B1),
                    group_rms(AdapterParameterGroup2d::B2),
                )
            };
        Ok(BurnE2eAdapterDiagnostics {
            parameterization: if dense_row_residual {
                E2E_HYPER_ADAPTER_DENSE_ROW_RESIDUAL
            } else {
                E2E_HYPER_ADAPTER_FACTORIZED
            },
            parameter_count,
            transport_parameter_count,
            dense_controller_dims,
            mean_l2_norm,
            min_l2_norm,
            max_l2_norm,
            mean_pairwise_l2_distance: (pairs > 0)
                .then_some(pairwise_distance_sum / pairs.max(1) as f32),
            min_pairwise_l2_distance: (pairs > 0).then_some(min_pairwise_l2_distance),
            mean_pairwise_cosine_similarity: (pairs > 0)
                .then_some(pairwise_cosine_sum / pairs.max(1) as f32),
            w1_dense_rms,
            w2_dense_rms,
            w1_down_rms: factor_rms.map(|values| values[0]),
            w1_up_rms: factor_rms.map(|values| values[1]),
            w2_down_rms: factor_rms.map(|values| values[2]),
            w2_up_rms: factor_rms.map(|values| values[3]),
            b1_rms,
            b2_rms,
        })
    }
