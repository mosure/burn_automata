use crate::cli::prelude::*;

pub(crate) fn run_bench(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::Bench {
        preset,
        model: model_path,
        particles,
        steps,
        repeats,
        update_prob,
        gpu,
        neighbor_mode,
        bucket_capacity,
        profile,
        seed_scale,
        normalize_seed_scale,
        fixed_eps,
        reference_seed_scale,
        seed_mode,
        geometry,
        gaussian,
        step_timing,
    } = command
    else {
        unreachable!("run_bench called with the wrong command variant");
    };

    #[cfg(not(feature = "gpu_wgpu"))]
    let _ = (
        neighbor_mode,
        bucket_capacity,
        gaussian,
        repeats,
        step_timing,
    );
    let preset: AutomataPreset = preset.into();
    let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
    let seed_mode: ParticleSeed = seed_mode.into();
    let normalize_seed_scale = normalize_seed_scale || !fixed_eps;
    let reference_seed_scale = reference_seed_scale
        .unwrap_or_else(|| reference_seed_scale_for_seed_mode(preset, seed_mode));
    let (model, base_grid) = if let Some(path) = model_path {
        let manifest = crate::import::load_manifest(&path)?;
        let hashgrid = manifest.hashgrid.clone();
        (manifest.into_model(), hashgrid)
    } else {
        let (config, hashgrid) = NpaConfig::for_preset(preset);
        (NpaModel::seeded(config, 42), hashgrid)
    };
    let grid = if normalize_seed_scale {
        model
            .config
            .hashgrid_for_seed_scale(&base_grid, seed_scale, reference_seed_scale)
    } else {
        base_grid
    };
    let start = Instant::now();
    if gpu {
        #[cfg(feature = "gpu_wgpu")]
        {
            let report = gpu_rollout_bench(
                &model,
                &grid,
                GpuBenchConfig {
                    particles,
                    steps,
                    seed_scale,
                    update_prob,
                    seed_mode,
                    geometry,
                    neighbor_mode: wgpu_neighbor_mode(neighbor_mode, bucket_capacity),
                    gaussian_write: gaussian,
                    step_timing,
                },
            )?;
            let reports = if repeats > 1 {
                let mut reports = Vec::with_capacity(repeats);
                reports.push(report);
                for _ in 1..repeats {
                    reports.push(gpu_rollout_bench(
                        &model,
                        &grid,
                        GpuBenchConfig {
                            particles,
                            steps,
                            seed_scale,
                            update_prob,
                            seed_mode,
                            geometry,
                            neighbor_mode: wgpu_neighbor_mode(neighbor_mode, bucket_capacity),
                            gaussian_write: gaussian,
                            step_timing,
                        },
                    )?);
                }
                reports
            } else {
                vec![report]
            };
            let summary = summarize_gpu_reports(&reports, steps);
            let report = summary.median_report;
            let avg_step_ms = report.gpu_step_ms / steps.max(1) as f64;
            let timing = if step_timing {
                "step_wait"
            } else {
                "submit_wait"
            };
            println!(
                "backend=wgpu particles={particles} steps={steps} repeats={} update_prob={update_prob:.3} geometry={geometry:?} elapsed_ms={:.6} gpu_step_ms={:.6} avg_step_ms={avg_step_ms:.6} min_avg_step_ms={:.6} median_avg_step_ms={:.6} max_avg_step_ms={:.6} step_timing={} step_min_ms={:.6} step_median_ms={:.6} step_p95_ms={:.6} step_p99_ms={:.6} step_max_ms={:.6} step_jitter_ratio={:.6} final_mean_displacement_per_step={:.6} final_mean_density={:.6} initial_nonempty_cells={} initial_max_cell_occupancy={} hashgrid=gpu-local hashgrid_eps={:.6} normalized_seed_scale={} reference_seed_scale={:.6} resident_state=true timing={timing} readback=final gaussian_write={} neighbor_mode={:?} bucket_capacity={} grid_storage_u32={} grid_clear_u32={} grid_overflow_count={} grid_max_overflow_count={} grid_overflowed_steps={}",
                summary.repeats,
                start.elapsed().as_secs_f64() * 1000.0,
                report.gpu_step_ms,
                summary.min_avg_step_ms,
                summary.median_avg_step_ms,
                summary.max_avg_step_ms,
                step_timing,
                report.step_min_ms,
                report.step_median_ms,
                report.step_p95_ms,
                report.step_p99_ms,
                report.step_max_ms,
                report.step_jitter_ratio,
                report.final_mean_dx,
                report.final_mean_density,
                report.initial_nonempty_cells,
                report.initial_max_cell_occupancy,
                grid.eps,
                normalize_seed_scale,
                reference_seed_scale,
                report.gaussian_write,
                report.neighbor_mode,
                report.bucket_capacity,
                report.grid_storage_len,
                report.grid_clear_len,
                report.grid_overflow_count,
                report.grid_max_overflow_count,
                report.grid_overflowed_steps
            );
        }
        #[cfg(not(feature = "gpu_wgpu"))]
        {
            return Err(std::io::Error::other(
                "bench --gpu requires building burn_automata with --features gpu_wgpu",
            )
            .into());
        }
    } else if profile {
        let profile = profile_rollout(
            &model,
            &grid,
            CpuProfileConfig {
                particles,
                steps,
                seed_scale,
                update_prob,
                seed_mode,
                geometry,
            },
        )?;
        println!(
            "particles={particles} steps={steps} update_prob={update_prob:.3} geometry={geometry:?} elapsed_ms={:.6} perceive_ms={:.6} forward_ms={:.6} integrate_ms={:.6} final_mean_dx={:.6}",
            start.elapsed().as_secs_f64() * 1000.0,
            profile.perceive_ms,
            profile.forward_ms,
            profile.integrate_ms,
            profile.final_mean_dx
        );
    } else {
        let trace = run_rollout(
            &model,
            &grid,
            &RolloutConfig {
                steps,
                particle_count: particles,
                update_prob,
                seed_scale,
                ..RolloutConfig::default()
            },
            seed_mode,
        )?;
        println!(
            "particles={particles} steps={steps} update_prob={update_prob:.3} elapsed_ms={} final_mean_dx={:.6}",
            start.elapsed().as_secs_f64() * 1000.0,
            trace.mean_dx.last().copied().unwrap_or_default()
        );
    }

    Ok(())
}

pub(crate) fn run_bench_spatial(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::BenchSpatial {
        preset,
        particles,
        seed_scale,
        normalize_seed_scale,
        fixed_eps,
        reference_seed_scale,
        seed_mode,
        geometry,
        strategy,
        bvh_leaf_size,
        tile_size,
    } = command
    else {
        unreachable!("run_bench_spatial called with the wrong command variant");
    };

    let preset: AutomataPreset = preset.into();
    let seed_scale = seed_scale.unwrap_or_else(|| NpaConfig::seed_scale_for_preset(preset));
    let seed_mode: ParticleSeed = seed_mode.into();
    let normalize_seed_scale = normalize_seed_scale || !fixed_eps;
    let reference_seed_scale = reference_seed_scale
        .unwrap_or_else(|| reference_seed_scale_for_seed_mode(preset, seed_mode));
    let (config, base_grid) = NpaConfig::for_preset(preset);
    let model = NpaModel::seeded(config.clone(), 42);
    let grid = if normalize_seed_scale {
        model
            .config
            .hashgrid_for_seed_scale(&base_grid, seed_scale, reference_seed_scale)
    } else {
        base_grid
    };
    let (positions, _states) = bench_particles(
        &model, &grid, particles, seed_scale, seed_mode, geometry, 42,
    );
    let strategies =
        spatial_strategies(strategy, &grid, parse_tile_size(&tile_size)?, bvh_leaf_size);
    for strategy in strategies {
        let started = Instant::now();
        match crate::kernels::analyze_spatial_strategy(&positions, 1, particles, &grid, strategy) {
            Ok(report) => {
                println!(
                    "backend=cpu-spatial preset={preset:?} particles={particles} geometry={geometry:?} strategy={} dim={} eps={:.6} analyze_ms={:.6} active_bins={} max_bin_occupancy={} candidates_per_particle={:.6} entries_per_particle={:.6} exact_neighbors_per_particle={:.6} node_visits_per_particle={:.6} node_count={} max_depth={} exact_neighbor_pairs={} candidate_tests={} candidate_entries_visited={}",
                    strategy_label(report.strategy),
                    report.dim,
                    report.eps,
                    started.elapsed().as_secs_f64() * 1000.0,
                    report.active_bins,
                    report.max_bin_occupancy,
                    report.candidates_per_particle(),
                    report.entries_per_particle(),
                    report.exact_neighbors_per_particle(),
                    report.node_visits_per_particle(),
                    report.node_count,
                    report.max_depth,
                    report.exact_neighbor_pairs,
                    report.candidate_tests,
                    report.candidate_entries_visited,
                );
            }
            Err(err) => {
                println!(
                    "backend=cpu-spatial preset={preset:?} particles={particles} geometry={geometry:?} strategy={} error=\"{}\"",
                    strategy_label(strategy),
                    err
                );
            }
        }
    }

    Ok(())
}
