use std::time::Instant;

use burn_automata::adaptive::{AdaptiveMaterialSeedLayout, AdaptiveTarget2dMaterialConfig};
use burn_automata::{
    AdaptiveExperimentConfig, AdaptiveLocalRuleSemantics, AdaptiveMultiscaleTrainingConfig,
    AdaptiveNpaConfig, AdaptiveNpaModel, AdaptiveParticleSet, AdaptiveReplayBackend,
    AdaptiveReplayTeacher, AdaptiveRolloutConfig, AdaptiveTopologyControl, AutomataPreset,
    NpaModel, NpaWeights, ParticleSeed, adaptive_deployment_on_policy_batch_wgpu,
    adaptive_isotropic_gaussian_geometry, adaptive_multiscale_on_policy_batch,
    evaluate_adaptive_task_quality_validation,
    gpu::{WgpuMaterialStateInit, WgpuSupportBinConfig},
    load_adaptive_model, material_footprint_radius,
    rollout::seed_particles_scaled,
    run_adaptive_rollout, seed_adaptive_particles_scaled,
    validate_adaptive_task_quality_validation_gates,
};
use burn_automata_kernels::AdaptiveGraphPolicy;

use crate::common::{max_abs_error, max_position_abs_error, new_executor_or_skip, wgpu_test_guard};

#[test]
fn resident_bootstrap_activates_canonical_rows_without_reallocation()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = wgpu_test_guard();
    let Some(executor) = new_executor_or_skip()? else {
        return Ok(());
    };
    let (rule_config, grid) = burn_automata::NpaConfig::for_preset(AutomataPreset::Growing2d);
    let rule = NpaModel {
        weights: NpaWeights::zeros(&rule_config),
        config: rule_config,
    };
    let mut adaptive = AdaptiveNpaConfig::growing_2d();
    adaptive.min_leaves = 16;
    adaptive.target_leaves = 40;
    adaptive.bootstrap_target_leaves = 40;
    adaptive.max_leaves = 64;
    adaptive.initial_leaves = 16;
    adaptive.bootstrap_fine_leaves = 64;
    adaptive.topology_interval = 1;
    adaptive.topology_start_step = 1;
    adaptive.bootstrap_end_step = 2;
    adaptive.bootstrap_events_per_interval = 4;
    adaptive.retain_bootstrap_templates = false;
    adaptive.runtime_topology_control = AdaptiveTopologyControl::PairedLocalDetail;
    adaptive.material_seed_bandwidth_exponent = 0.5;
    adaptive.perception.min_bandwidth = 0.1;
    adaptive.perception.max_bandwidth = 0.2;
    adaptive.perception.support_bin_ratio = 2.0;
    let model = AdaptiveNpaModel::seeded(rule, adaptive, 19)?;
    let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
    let particles = seed_adaptive_particles_scaled(
        &model,
        16,
        13,
        ParticleSeed::UniformCircle,
        0.2,
        total_measure,
        0.1,
    )?;
    assert!(particles.bootstrap_templates.is_empty());
    let initial_measure = particles.total_measure();
    let mut gpu = executor.create_adaptive_state(
        &model,
        particles,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::CooperativeSortedCells,
        1.0,
        11,
    )?;
    assert_eq!(gpu.resident.particle_count, 16);
    assert_eq!(gpu.resident.particle_capacity, 40);

    let report = executor.step_adaptive_state_many(&mut gpu, 2, true)?;
    assert_eq!(report.resident_particle_count, 40);
    assert_eq!(gpu.resident.particle_count, 40);
    assert_eq!(gpu.resident.particle_capacity, 40);
    assert_eq!(report.topology_updates.len(), 2);
    assert_eq!(
        report
            .topology_updates
            .iter()
            .map(|update| update.split_events)
            .sum::<usize>(),
        8
    );
    assert_eq!(
        executor.read_adaptive_local_detail_topology_accept_count(&gpu)?,
        8
    );
    executor.synchronize_adaptive_particles(&mut gpu)?;
    assert_eq!(gpu.particles.len(), 40);
    assert!((gpu.particles.total_measure() - initial_measure).abs() <= 1.0e-8);
    let min_measure = gpu
        .particles
        .represented_measure
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_measure = gpu
        .particles
        .represented_measure
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    assert!((max_measure / min_measure - 4.0).abs() <= 1.0e-5);
    Ok(())
}

#[test]
fn paired_local_detail_topology_matches_cpu_without_runtime_readback()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = wgpu_test_guard();
    let Some(executor) = new_executor_or_skip()? else {
        return Ok(());
    };
    let (rule_config, grid) = burn_automata::NpaConfig::for_preset(AutomataPreset::Growing2d);
    let state_dims = rule_config.state_dims;
    let rule = NpaModel {
        weights: NpaWeights::zeros(&rule_config),
        config: rule_config,
    };
    let positions = vec![
        [0.18, 0.12, 0.0, 0.0],
        [-0.22, -0.20, 0.0, 0.0],
        [-0.20, -0.22, 0.0, 0.0],
        [-0.18, -0.20, 0.0, 0.0],
        [-0.20, -0.18, 0.0, 0.0],
        [0.16, 0.12, 0.0, 0.0],
        [0.24, 0.20, 0.0, 0.0],
        [0.20, 0.28, 0.0, 0.0],
    ];
    let states = (0..positions.len())
        .flat_map(|row| {
            (0..state_dims).map(move |channel| match row {
                5 => 8.0 + channel as f32 * 0.1,
                6 => -8.0 - channel as f32 * 0.1,
                7 => 4.0 + channel as f32 * 0.05,
                _ => 0.0,
            })
        })
        .collect::<Vec<_>>();
    let fine_measure = 0.0025;
    let fine_footprint = material_footprint_radius(fine_measure, 2);
    let mut particles = AdaptiveParticleSet::from_equal_measure(
        positions,
        states,
        2,
        state_dims,
        fine_measure * 8.0,
        0.4,
    )?;
    particles.represented_measure[0] = 4.0 * fine_measure;
    for row in 0..particles.len() {
        let footprint = particles.footprint(row);
        particles.render_footprint[row] = footprint;
        particles.bandwidth[row] = 0.4;
        let variance = (0.5 * footprint).powi(2);
        particles.covariance[row] = [variance, 0.0, 0.0, 0.0, variance, 0.0, 0.0, 0.0, variance];
    }
    particles.validate()?;
    let initial_particles = particles.clone();

    let mut adaptive = AdaptiveNpaConfig::growing_2d();
    adaptive.reference_footprint = fine_footprint;
    adaptive.base_rule_footprint = fine_footprint;
    adaptive.min_footprint = fine_footprint * 0.5;
    adaptive.max_footprint = fine_footprint * 4.0;
    adaptive.min_leaves = particles.len();
    adaptive.target_leaves = particles.len();
    adaptive.max_leaves = particles.len();
    adaptive.hierarchical_bootstrap_seed = false;
    adaptive.retain_bootstrap_templates = false;
    adaptive.topology_interval = 1;
    adaptive.steady_topology_interval = 1;
    adaptive.topology_start_step = 1;
    adaptive.steady_topology_start_step = 1;
    adaptive.runtime_topology_control = AdaptiveTopologyControl::PairedLocalDetail;
    adaptive.max_events_per_interval = 1;
    adaptive.proxy.enabled = false;
    adaptive.rule_graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive.perception.graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive.perception.min_bandwidth = 0.4;
    adaptive.perception.max_bandwidth = 0.4;
    let model = AdaptiveNpaModel::seeded(rule, adaptive, 19)?;

    let cpu = run_adaptive_rollout(
        &model,
        particles.clone(),
        AdaptiveRolloutConfig {
            steps: 4,
            dt: 1.0,
            update_prob: 1.0,
            seed: 11,
            bandwidth_adaptation_enabled: false,
            topology_enabled: true,
            snapshot_interval: 1,
        },
    )?;
    assert_eq!(cpu.metrics.len(), 4);
    let cpu_accepts = cpu
        .metrics
        .iter()
        .map(|metric| metric.split_events)
        .sum::<usize>();
    assert!(cpu_accepts > 0);
    assert!(
        cpu.metrics
            .iter()
            .all(|metric| metric.split_events == metric.merge_events)
    );

    let mut gpu = executor.create_adaptive_state(
        &model,
        particles,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::CooperativeSortedCells,
        1.0,
        11,
    )?;
    let report = executor.step_adaptive_state_many(&mut gpu, 4, true)?;
    assert_eq!(
        executor.read_adaptive_local_detail_topology_accept_count(&gpu)?,
        cpu_accepts
    );
    assert_eq!(report.topology_updates.len(), 4);
    assert_eq!(report.particle_steps, 32);
    assert_eq!(report.topology_particle_steps, 0);
    assert_eq!(report.interaction_particle_steps, 32);
    assert!(
        report
            .topology_updates
            .iter()
            .all(|update| (update.split_events, update.merge_events) == (1, 1))
    );
    executor.synchronize_adaptive_particles(&mut gpu)?;

    let position_error = max_position_abs_error(&cpu.particles.positions, &gpu.particles.positions);
    let physical_position_error =
        max_position_set_abs_error(&cpu.particles.positions, &gpu.particles.positions);
    let state_error = max_abs_error(&cpu.particles.states, &gpu.particles.states);
    eprintln!(
        "paired topology parity: physical_position={physical_position_error:.7} \
         row_position={position_error:.7} state={state_error:.7}"
    );
    assert!(
        physical_position_error <= 3.0e-3,
        "physical position error {physical_position_error}"
    );
    assert!(state_error <= 3.0e-3, "state error {state_error}");

    let mut rejected_model = model.clone();
    rejected_model.config.min_reallocation_relative_gain = 1.0;
    rejected_model.validate()?;
    let rejected_cpu = run_adaptive_rollout(
        &rejected_model,
        initial_particles.clone(),
        AdaptiveRolloutConfig {
            steps: 4,
            dt: 1.0,
            update_prob: 1.0,
            seed: 11,
            bandwidth_adaptation_enabled: false,
            topology_enabled: true,
            snapshot_interval: 1,
        },
    )?;
    assert!(
        rejected_cpu
            .metrics
            .iter()
            .all(|metric| (metric.split_events, metric.merge_events) == (0, 0))
    );
    let mut rejected_gpu = executor.create_adaptive_state(
        &rejected_model,
        initial_particles,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::CooperativeSortedCells,
        1.0,
        11,
    )?;
    executor.step_adaptive_state_many(&mut rejected_gpu, 4, true)?;
    assert_eq!(
        executor.read_adaptive_local_detail_topology_accept_count(&rejected_gpu)?,
        0
    );
    executor.synchronize_adaptive_particles(&mut rejected_gpu)?;
    assert!(
        max_position_abs_error(
            &rejected_gpu.particles.positions,
            &rejected_cpu.particles.positions
        ) <= 1.0e-6
    );
    assert!(
        max_abs_error(
            &rejected_gpu.particles.states,
            &rejected_cpu.particles.states
        ) <= 1.0e-6
    );
    assert_eq!(
        rejected_gpu.particles.represented_measure,
        rejected_cpu.particles.represented_measure
    );
    Ok(())
}

#[test]
fn continuous_local_detail_topology_matches_cpu_without_runtime_readback()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = wgpu_test_guard();
    let Some(executor) = new_executor_or_skip()? else {
        return Ok(());
    };
    let (rule_config, grid) = burn_automata::NpaConfig::for_preset(AutomataPreset::Growing2d);
    let state_dims = rule_config.state_dims;
    let rule = NpaModel {
        weights: NpaWeights::zeros(&rule_config),
        config: rule_config,
    };
    let positions = vec![
        [-0.50, -0.15, 0.0, 0.0],
        [-0.32, 0.08, 0.0, 0.0],
        [-0.14, -0.06, 0.0, 0.0],
        [-0.80, -0.80, 0.0, 0.0],
        [0.16, -0.10, 0.0, 0.0],
        [0.82, 0.75, 0.0, 0.0],
        [0.36, -0.02, 0.0, 0.0],
        [0.42, 0.04, 0.0, 0.0],
    ];
    let total_measure = 0.02;
    let material = AdaptiveTarget2dMaterialConfig {
        reference_particle_count: positions.len(),
        total_measure,
        fine_bandwidth: 0.4,
        bandwidth_exponent: 0.0,
        max_initial_fine_units: 4,
        seed_layout: AdaptiveMaterialSeedLayout::GradedContinuous,
        seed_measure_ratio: 1.44,
    }
    .layout(positions.len(), 0.01, 0.4)?;
    let coarse_row = material
        .represented_measure
        .iter()
        .enumerate()
        .max_by(|lhs, rhs| lhs.1.total_cmp(rhs.1))
        .map(|(row, _)| row)
        .unwrap();
    let states = (0..positions.len())
        .flat_map(|row| {
            (0..state_dims).map(move |channel| {
                if row == coarse_row {
                    20.0 + channel as f32 * 0.25
                } else {
                    0.0
                }
            })
        })
        .collect::<Vec<_>>();
    let mut particles = AdaptiveParticleSet::from_equal_measure(
        positions,
        states,
        2,
        state_dims,
        total_measure,
        0.4,
    )?;
    particles.represented_measure = material.represented_measure;
    particles.bandwidth = material.bandwidth;
    for row in 0..particles.len() {
        let footprint = particles.footprint(row);
        particles.render_footprint[row] = footprint;
        let variance = (0.5 * footprint).powi(2);
        particles.covariance[row] = [variance, 0.0, 0.0, 0.0, variance, 0.0, 0.0, 0.0, variance];
    }
    particles.validate()?;
    let initial_particles = particles.clone();

    let fine_footprint = material_footprint_radius(total_measure / particles.len() as f32, 2);
    let mut adaptive = AdaptiveNpaConfig::growing_2d();
    adaptive.material_seed_layout = AdaptiveMaterialSeedLayout::GradedContinuous;
    adaptive.material_seed_measure_ratio = 1.44;
    adaptive.material_seed_bandwidth_exponent = 0.0;
    adaptive.reference_footprint = fine_footprint;
    adaptive.base_rule_footprint = fine_footprint;
    adaptive.min_footprint = fine_footprint * 0.5;
    adaptive.max_footprint = fine_footprint * 2.0;
    adaptive.min_leaves = particles.len();
    adaptive.target_leaves = particles.len();
    adaptive.max_leaves = particles.len();
    adaptive.bootstrap_fine_leaves = particles.len();
    adaptive.hierarchical_bootstrap_seed = false;
    adaptive.retain_bootstrap_templates = false;
    adaptive.topology_interval = 1;
    adaptive.steady_topology_interval = 1;
    adaptive.topology_start_step = 1;
    adaptive.steady_topology_start_step = 1;
    adaptive.runtime_topology_control = AdaptiveTopologyControl::ContinuousLocalDetail;
    adaptive.max_events_per_interval = 2;
    adaptive.proxy.enabled = false;
    adaptive.rule_graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive.perception.graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive.perception.min_bandwidth = 0.4;
    adaptive.perception.max_bandwidth = 0.4;
    let model = AdaptiveNpaModel::seeded(rule, adaptive, 19)?;

    let rollout = AdaptiveRolloutConfig {
        steps: 1,
        dt: 1.0,
        update_prob: 1.0,
        seed: 11,
        bandwidth_adaptation_enabled: false,
        topology_enabled: true,
        snapshot_interval: 1,
    };
    let cpu = run_adaptive_rollout(&model, particles.clone(), rollout)?;
    let cpu_accepts = cpu
        .metrics
        .iter()
        .map(|metric| metric.split_events)
        .sum::<usize>();
    assert_eq!(cpu_accepts, 2);

    let weighted = |particles: &AdaptiveParticleSet, values: &[f32], width: usize| {
        let mut sum = vec![0.0_f64; width];
        for row in 0..particles.len() {
            for channel in 0..width {
                sum[channel] += values[row * width + channel] as f64
                    * particles.represented_measure[row] as f64;
            }
        }
        sum
    };
    let spatial_second_moment = |particles: &AdaptiveParticleSet| {
        let mut moment = [0.0_f64; 3];
        for row in 0..particles.len() {
            let weight = f64::from(particles.represented_measure[row]);
            let x = f64::from(particles.positions[row][0]);
            let y = f64::from(particles.positions[row][1]);
            moment[0] += weight * (f64::from(particles.covariance[row][0]) + x * x);
            moment[1] += weight * (f64::from(particles.covariance[row][1]) + x * y);
            moment[2] += weight * (f64::from(particles.covariance[row][4]) + y * y);
        }
        moment
    };
    let initial_positions = initial_particles
        .positions
        .iter()
        .flat_map(|position| [position[0], position[1]])
        .collect::<Vec<_>>();
    let final_positions = cpu
        .particles
        .positions
        .iter()
        .flat_map(|position| [position[0], position[1]])
        .collect::<Vec<_>>();
    assert!(
        max_abs_error(
            &weighted(&initial_particles, &initial_positions, 2)
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
            &weighted(&cpu.particles, &final_positions, 2)
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
        ) <= 2.0e-6
    );
    assert!(
        max_abs_error(
            &spatial_second_moment(&initial_particles)
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
            &spatial_second_moment(&cpu.particles)
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
        ) <= 2.0e-6
    );
    assert!(
        max_abs_error(
            &weighted(&initial_particles, &initial_particles.states, state_dims)
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
            &weighted(&cpu.particles, &cpu.particles.states, state_dims)
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
        ) <= 2.0e-6
    );

    let mut gpu = executor.create_adaptive_state(
        &model,
        particles,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::CooperativeSortedCells,
        1.0,
        11,
    )?;
    let report = executor.step_adaptive_state_many(&mut gpu, 1, true)?;
    assert_eq!(
        executor.read_adaptive_local_detail_topology_accept_count(&gpu)?,
        cpu_accepts
    );
    assert_eq!(report.particle_steps, 8);
    assert_eq!(report.topology_particle_steps, 0);
    assert_eq!(report.topology_updates.len(), 1);
    assert_eq!(
        (
            report.topology_updates[0].split_events,
            report.topology_updates[0].merge_events,
        ),
        (2, 2),
    );
    executor.synchronize_adaptive_particles(&mut gpu)?;
    let position_error = max_position_abs_error(&cpu.particles.positions, &gpu.particles.positions);
    let state_error = max_abs_error(&cpu.particles.states, &gpu.particles.states);
    eprintln!("continuous topology parity: position={position_error:.7} state={state_error:.7}");
    if position_error > 3.0e-3 {
        eprintln!("continuous CPU positions={:?}", cpu.particles.positions);
        eprintln!("continuous GPU positions={:?}", gpu.particles.positions);
        eprintln!(
            "continuous represented measure={:?}",
            cpu.particles.represented_measure
        );
    }
    assert!(position_error <= 3.0e-3, "position error {position_error}");
    assert!(state_error <= 3.0e-3, "state error {state_error}");
    assert!(
        max_abs_error(
            &spatial_second_moment(&initial_particles)
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
            &spatial_second_moment(&gpu.particles)
                .into_iter()
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
        ) <= 3.0e-5
    );
    assert_eq!(
        gpu.particles.represented_measure,
        cpu.particles.represented_measure
    );
    Ok(())
}

fn max_position_set_abs_error(lhs: &[[f32; 4]], rhs: &[[f32; 4]]) -> f32 {
    let mut lhs = lhs.to_vec();
    let mut rhs = rhs.to_vec();
    let compare = |a: &[f32; 4], b: &[f32; 4]| {
        a.iter()
            .zip(b)
            .find_map(|(a, b)| {
                let ordering = a.total_cmp(b);
                ordering.is_ne().then_some(ordering)
            })
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    lhs.sort_by(compare);
    rhs.sort_by(compare);
    max_position_abs_error(&lhs, &rhs)
}

#[test]
fn represented_measure_wgpu_step_matches_adaptive_cpu_oracle()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = wgpu_test_guard();
    let Some(executor) = new_executor_or_skip()? else {
        return Ok(());
    };
    let particles = 192;
    let (rule_config, grid) = burn_automata::NpaConfig::for_preset(AutomataPreset::Growing2d);
    let rule = burn_automata::NpaModel::seeded(rule_config, 17);
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        rule.config.state_dims,
        rule.config.spatial_dims,
        43,
        ParticleSeed::UniformCircle,
        0.2,
    );
    let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
    let mut particle_set = AdaptiveParticleSet::from_equal_measure(
        positions,
        states,
        2,
        rule.config.state_dims,
        total_measure,
        grid.eps,
    )?;
    let raw_weights = (0..particles)
        .map(|index| 0.55 + 0.9 * ((index * 37 % particles) as f32 / particles as f32))
        .collect::<Vec<_>>();
    let weight_sum = raw_weights.iter().sum::<f32>();
    for (index, weight) in raw_weights.into_iter().enumerate() {
        let measure = total_measure * weight / weight_sum;
        particle_set.represented_measure[index] = measure;
        let footprint = material_footprint_radius(measure, 2);
        particle_set.render_footprint[index] = footprint;
        let variance = (0.5 * footprint).powi(2);
        particle_set.covariance[index] =
            [variance, 0.0, 0.0, 0.0, variance, 0.0, 0.0, 0.0, variance];
        particle_set.bandwidth[index] = grid.eps;
    }
    particle_set.validate()?;

    let mut adaptive_config = AdaptiveNpaConfig::growing_2d();
    adaptive_config.reference_footprint =
        material_footprint_radius(total_measure / particles as f32, 2);
    adaptive_config.base_rule_footprint = adaptive_config.reference_footprint;
    adaptive_config.min_footprint = particle_set
        .represented_measure
        .iter()
        .map(|measure| material_footprint_radius(*measure, 2))
        .fold(f32::INFINITY, f32::min)
        * 0.5;
    adaptive_config.max_footprint = particle_set
        .represented_measure
        .iter()
        .map(|measure| material_footprint_radius(*measure, 2))
        .fold(0.0_f32, f32::max)
        * 2.0;
    adaptive_config.min_leaves = particles;
    adaptive_config.target_leaves = particles;
    adaptive_config.max_leaves = particles;
    adaptive_config.rule_graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive_config.proxy.enabled = false;
    adaptive_config.perception.graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive_config.perception.reference_measure = total_measure / particles as f32;
    adaptive_config.perception.min_bandwidth = grid.eps;
    adaptive_config.perception.max_bandwidth = grid.eps;
    adaptive_config.perception.support_bin_ratio = 2.0_f32.sqrt();
    let mut adaptive_model = AdaptiveNpaModel::seeded(rule, adaptive_config, 19)?;
    adaptive_model.enable_material_scale_conditioning()?;
    adaptive_model.enable_material_conditioned_compatible_residual_rule()?;
    adaptive_model
        .local_residual_rule
        .as_mut()
        .unwrap()
        .weights
        .b2[0] = 0.05;
    let input_dims = adaptive_model.rule.config.perception_dims();
    for hidden in 0..adaptive_model.rule.config.hidden_dims {
        adaptive_model.rule.weights.w1[hidden * input_dims + input_dims - 1] =
            (hidden as f32 + 1.0) * 1.0e-4;
    }

    let cpu = run_adaptive_rollout(
        &adaptive_model,
        particle_set.clone(),
        AdaptiveRolloutConfig {
            steps: 1,
            dt: 1.0,
            update_prob: 1.0,
            seed: 11,
            bandwidth_adaptation_enabled: false,
            topology_enabled: false,
            snapshot_interval: 1,
        },
    )?;
    let mut gpu = executor.create_adaptive_state(
        &adaptive_model,
        particle_set,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::CooperativeSortedCells,
        1.0,
        11,
    )?;
    executor.step_adaptive_state_many(&mut gpu, 1, false)?;
    executor.synchronize_adaptive_particles(&mut gpu)?;

    let position_error = max_position_abs_error(&cpu.particles.positions, &gpu.particles.positions);
    let state_error = max_abs_error(&cpu.particles.states, &gpu.particles.states);
    eprintln!("represented-measure parity: position={position_error:.7} state={state_error:.7}");
    assert!(position_error <= 3.0e-3, "position error {position_error}");
    assert!(state_error <= 3.0e-3, "state error {state_error}");
    Ok(())
}

#[test]
fn adaptive_support_bin_policy_refreshes_without_reallocation()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = wgpu_test_guard();
    let Some(executor) = new_executor_or_skip()? else {
        return Ok(());
    };
    const PARTICLES: usize = 4_096;
    let (config, grid) = burn_automata::NpaConfig::for_preset(AutomataPreset::Growing2d);
    let model = NpaModel {
        weights: NpaWeights::zeros(&config),
        config,
    };
    let (positions, states) = seed_particles_scaled(
        1,
        PARTICLES,
        model.config.state_dims,
        2,
        271,
        ParticleSeed::UniformCircle,
        0.2,
    );
    let represented_measure = vec![1.0 / PARTICLES as f32; PARTICLES];
    let initial_bandwidth = vec![0.025; PARTICLES];
    let fine_heavy_bandwidth = (0..PARTICLES)
        .map(|index| if index < PARTICLES / 10 { 0.2 } else { 0.025 })
        .collect::<Vec<_>>();
    let balanced_bandwidth = (0..PARTICLES)
        .map(|index| {
            let fraction = (index * 977 % PARTICLES) as f32 / (PARTICLES - 1) as f32;
            0.025 * 8.0_f32.powf(fraction)
        })
        .collect::<Vec<_>>();
    let covariance = vec![[0.0; 9]; PARTICLES];
    let state_jacobian = vec![0.0; PARTICLES * model.config.state_dims * 2];
    let render_scale = vec![1.0; PARTICLES];
    let support_bins = Some(WgpuSupportBinConfig {
        min_bandwidth: 0.025,
        max_bandwidth: 0.2,
        ratio: 2.0,
        force: false,
    });
    let neighbor_mode = if executor.subgroup_cooperative_supported() {
        burn_automata::gpu::WgpuNeighborMode::SubgroupCooperativeSortedCells
    } else {
        burn_automata::gpu::WgpuNeighborMode::CooperativeSortedCells
    };
    let mut resident = executor.create_material_state_with_neighbor_mode_and_update_prob(
        &model,
        &positions,
        &states,
        1,
        PARTICLES,
        &grid,
        1.0,
        neighbor_mode,
        1.0,
        271,
        WgpuMaterialStateInit {
            represented_measure: &represented_measure,
            particle_ids: None,
            update_masks: None,
            bandwidth: &initial_bandwidth,
            support_bins,
            covariance: &covariance,
            state_jacobian: &state_jacobian,
            closure_mode: None,
            closure_basis: None,
            closure_phase: None,
            render_from_scale: &render_scale,
            render_target_footprint: &render_scale,
            display_scale_per_footprint: 1.0,
            render_transition_steps: 0,
        },
    )?;
    let initial = executor.neighbor_report(&resident);
    assert_eq!(initial.requested_support_bin_count, 3);
    assert_eq!(initial.support_bin_capacity, 3);
    assert_eq!(initial.support_bin_count, 1);

    executor.update_state_material_with_support_policy(
        &mut resident,
        &positions,
        &grid,
        WgpuMaterialStateInit {
            represented_measure: &represented_measure,
            particle_ids: None,
            update_masks: None,
            bandwidth: &fine_heavy_bandwidth,
            support_bins,
            covariance: &covariance,
            state_jacobian: &state_jacobian,
            closure_mode: None,
            closure_basis: None,
            closure_phase: None,
            render_from_scale: &render_scale,
            render_target_footprint: &render_scale,
            display_scale_per_footprint: 1.0,
            render_transition_steps: 0,
        },
    )?;
    assert_eq!(executor.neighbor_report(&resident).support_bin_count, 3);
    executor.step_state(&mut resident)?;

    executor.update_state_material_with_support_policy(
        &mut resident,
        &positions,
        &grid,
        WgpuMaterialStateInit {
            represented_measure: &represented_measure,
            particle_ids: None,
            update_masks: None,
            bandwidth: &balanced_bandwidth,
            support_bins,
            covariance: &covariance,
            state_jacobian: &state_jacobian,
            closure_mode: None,
            closure_basis: None,
            closure_phase: None,
            render_from_scale: &render_scale,
            render_target_footprint: &render_scale,
            display_scale_per_footprint: 1.0,
            render_transition_steps: 0,
        },
    )?;
    assert_eq!(executor.neighbor_report(&resident).support_bin_count, 1);
    executor.step_state(&mut resident)?;
    let output = executor.read_state(&resident)?;
    assert!(
        output
            .next_positions
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
    assert!(output.next_states.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
#[ignore = "device benchmark; run explicitly with --ignored --nocapture"]
fn adaptive_support_bin_wgpu_throughput_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = wgpu_test_guard();
    let Some(executor) = new_executor_or_skip()? else {
        return Ok(());
    };
    const PARTICLES: usize = 4_096;
    const WARMUP_STEPS: usize = 32;
    const SAMPLE_STEPS: usize = 128;
    const SAMPLES: usize = 7;
    let (config, grid) = burn_automata::NpaConfig::for_preset(AutomataPreset::Growing2d);
    let model = NpaModel {
        weights: NpaWeights::zeros(&config),
        config,
    };
    let distribution = std::env::var("BURN_AUTOMATA_SUPPORT_BIN_DISTRIBUTION")
        .unwrap_or_else(|_| "uniform-log".to_string());
    let seed_radius = if distribution.starts_with("clustered") {
        0.2
    } else {
        0.8
    };
    let (positions, states) = seed_particles_scaled(
        1,
        PARTICLES,
        model.config.state_dims,
        2,
        271,
        ParticleSeed::UniformCircle,
        seed_radius,
    );
    let represented_measure = vec![1.0 / PARTICLES as f32; PARTICLES];
    let bandwidth = (0..PARTICLES)
        .map(|index| match distribution.as_str() {
            "fine90" | "clustered-fine90" => {
                if index * 10 / PARTICLES == 0 {
                    0.2
                } else {
                    0.025
                }
            }
            _ => {
                let fraction = (index * 977 % PARTICLES) as f32 / (PARTICLES - 1) as f32;
                0.025 * 8.0_f32.powf(fraction)
            }
        })
        .collect::<Vec<_>>();
    let covariance = vec![[0.0; 9]; PARTICLES];
    let state_jacobian = vec![0.0; PARTICLES * model.config.state_dims * 2];
    let render_scale = vec![1.0; PARTICLES];
    let settings = [
        ("global", None),
        (
            "auto",
            Some(WgpuSupportBinConfig {
                min_bandwidth: 0.025,
                max_bandwidth: 0.2,
                ratio: 2.0,
                force: false,
            }),
        ),
        (
            "ratio2",
            Some(WgpuSupportBinConfig {
                min_bandwidth: 0.025,
                max_bandwidth: 0.2,
                ratio: 2.0,
                force: true,
            }),
        ),
        (
            "sqrt8",
            Some(WgpuSupportBinConfig {
                min_bandwidth: 0.025,
                max_bandwidth: 0.2,
                ratio: 8.0_f32.sqrt(),
                force: true,
            }),
        ),
        (
            "sqrt2",
            Some(WgpuSupportBinConfig {
                min_bandwidth: 0.025,
                max_bandwidth: 0.2,
                ratio: 2.0_f32.sqrt(),
                force: true,
            }),
        ),
    ];
    let mut cases = Vec::new();
    for (name, support_bins) in settings {
        let mut resident = executor.create_material_state_with_neighbor_mode_and_update_prob(
            &model,
            &positions,
            &states,
            1,
            PARTICLES,
            &grid,
            1.0,
            burn_automata::gpu::WgpuNeighborMode::SubgroupCooperativeSortedCells,
            1.0,
            271,
            WgpuMaterialStateInit {
                represented_measure: &represented_measure,
                particle_ids: None,
                update_masks: None,
                bandwidth: &bandwidth,
                support_bins,
                covariance: &covariance,
                state_jacobian: &state_jacobian,
                closure_mode: None,
                closure_basis: None,
                closure_phase: None,
                render_from_scale: &render_scale,
                render_target_footprint: &render_scale,
                display_scale_per_footprint: 1.0,
                render_transition_steps: 0,
            },
        )?;
        let report = executor.neighbor_report(&resident);
        if name == "auto" {
            let expected = usize::from(distribution == "clustered-fine90") * 2 + 1;
            assert_eq!(report.support_bin_count, expected);
        }
        executor.step_state_many(&mut resident, WARMUP_STEPS)?;
        let _ = executor.read_state(&resident)?;
        cases.push((name, report, resident, Vec::<f64>::with_capacity(SAMPLES)));
    }
    for sample in 0..SAMPLES {
        for offset in 0..cases.len() {
            let index = (sample + offset) % cases.len();
            let (_, _, resident, samples) = &mut cases[index];
            let started = Instant::now();
            executor.step_state_many(resident, SAMPLE_STEPS)?;
            let _ = executor.read_state(resident)?;
            samples.push(started.elapsed().as_secs_f64() * 1_000.0 / SAMPLE_STEPS as f64);
        }
    }

    let mut reference: Option<(Vec<[f32; 4]>, Vec<f32>)> = None;
    let mut measurements = Vec::new();
    for (name, report, resident, mut samples) in cases {
        let output = executor.read_state(&resident)?;
        if let Some((reference_positions, reference_states)) = &reference {
            assert!(max_position_abs_error(reference_positions, &output.next_positions) <= 1.0e-6);
            assert!(max_abs_error(reference_states, &output.next_states) <= 1.0e-6);
        } else {
            reference = Some((output.next_positions.clone(), output.next_states.clone()));
        }
        samples.sort_by(f64::total_cmp);
        let p50_ms = samples[samples.len() / 2];
        let p95_index = (samples.len() * 95)
            .div_ceil(100)
            .saturating_sub(1)
            .min(samples.len() - 1);
        let p95_ms = samples[p95_index];
        measurements.push((
            name,
            report.support_bin_count,
            report.grid_storage_len,
            p50_ms,
            p95_ms,
            PARTICLES as f64 * 1_000.0 / p50_ms,
        ));
    }
    let global_ms = measurements[0].3;
    let global_p95_ms = measurements[0].4;
    for (name, bins, storage, p50_ms, p95_ms, particle_steps_per_second) in measurements {
        eprintln!(
            "adaptive-support-bin backend=subgroup particles={PARTICLES} range=8x distribution={distribution} mode={name} bins={bins} storage_u32={storage} p50_step_ms={p50_ms:.4} p95_step_ms={p95_ms:.4} particle_steps_per_second={particle_steps_per_second:.0} p50_speedup={:.3}x p95_speedup={:.3}x",
            global_ms / p50_ms,
            global_p95_ms / p95_ms,
        );
    }
    Ok(())
}

#[test]
fn fused_closure_residual_wgpu_step_matches_adaptive_cpu_oracle()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = wgpu_test_guard();
    let Some(executor) = new_executor_or_skip()? else {
        return Ok(());
    };
    let particles = 192;
    let (rule_config, grid) = burn_automata::NpaConfig::for_preset(AutomataPreset::Growing2d);
    let rule = burn_automata::NpaModel::seeded(rule_config.clone(), 117);
    let mut local_config = rule_config;
    local_config.auxiliary_input_dims = 1 + 3 + local_config.state_dims * 2;
    let local = burn_automata::NpaModel::seeded(local_config, 211);
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        rule.config.state_dims,
        2,
        43,
        ParticleSeed::UniformCircle,
        0.05,
    );
    let total_measure = std::f32::consts::PI * 0.05_f32.powi(2);
    let mut particle_set = AdaptiveParticleSet::from_equal_measure(
        positions,
        states,
        2,
        rule.config.state_dims,
        total_measure,
        grid.eps,
    )?;
    for index in 0..particles {
        let phase = index as f32 * 0.137;
        particle_set.covariance[index] = [
            0.35 * particle_set.represented_measure[index],
            0.07 * phase.sin() * particle_set.represented_measure[index],
            0.0,
            0.07 * phase.sin() * particle_set.represented_measure[index],
            0.55 * particle_set.represented_measure[index],
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        for component in 0..particle_set.state_dims * particle_set.spatial_dims {
            particle_set.state_jacobian
                [index * particle_set.state_dims * particle_set.spatial_dims + component] =
                0.4 * (phase + component as f32 * 0.19).sin();
        }
    }
    let footprint = material_footprint_radius(total_measure / particles as f32, 2);
    let mut adaptive_config = AdaptiveNpaConfig::growing_2d();
    adaptive_config.reference_footprint = footprint;
    adaptive_config.base_rule_footprint = footprint * 0.8;
    adaptive_config.local_residual_scale = 0.35;
    adaptive_config.closure_moment_features = true;
    adaptive_config.min_footprint = footprint * 0.5;
    adaptive_config.max_footprint = footprint * 2.0;
    adaptive_config.min_leaves = particles;
    adaptive_config.target_leaves = particles;
    adaptive_config.max_leaves = particles;
    adaptive_config.proxy.enabled = false;
    adaptive_config.perception.graph_policy = AdaptiveGraphPolicy::DirectedTopK;
    adaptive_config.perception.max_neighbors = 128;
    adaptive_config.rule_graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive_config.perception.min_bandwidth = grid.eps;
    adaptive_config.perception.max_bandwidth = grid.eps;
    let mut model = AdaptiveNpaModel::seeded(rule, adaptive_config, 19)?;
    model.local_residual_rule = Some(local);
    model.validate()?;

    let cpu = run_adaptive_rollout(
        &model,
        particle_set.clone(),
        AdaptiveRolloutConfig {
            steps: 1,
            dt: 1.0,
            update_prob: 1.0,
            seed: 11,
            bandwidth_adaptation_enabled: false,
            topology_enabled: false,
            snapshot_interval: 1,
        },
    )?;
    let mut gpu = executor.create_adaptive_state(
        &model,
        particle_set,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::CooperativeSortedCells,
        1.0,
        11,
    )?;
    let buffers = executor.create_gaussian_buffers(particles)?;
    let bind_group = executor.create_gaussian_bind_group(&buffers.refs(), particles)?;
    executor.step_adaptive_state_many_into_gaussian_bind_group(&mut gpu, &bind_group, 1, false)?;
    let gpu = executor.read_state(&gpu.resident)?;

    let position_error = max_position_abs_error(&cpu.particles.positions, &gpu.next_positions);
    let state_error = max_abs_error(&cpu.particles.states, &gpu.next_states);
    eprintln!("fused adaptive closure parity: position={position_error:.7} state={state_error:.7}");
    assert!(position_error <= 3.0e-3, "position error {position_error}");
    assert!(state_error <= 3.0e-3, "state error {state_error}");
    Ok(())
}

#[test]
fn recurrent_closure_mode_wgpu_matches_cpu_across_multiple_steps()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = wgpu_test_guard();
    let Some(executor) = new_executor_or_skip()? else {
        return Ok(());
    };
    let particles = 64;
    let (rule_config, grid) = burn_automata::NpaConfig::for_preset(AutomataPreset::Growing2d);
    let rule = NpaModel::seeded(rule_config, 307);
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        rule.config.state_dims,
        2,
        311,
        ParticleSeed::UniformCircle,
        0.08,
    );
    let total_measure = std::f32::consts::PI * 0.08_f32.powi(2);
    let mut particle_set = AdaptiveParticleSet::from_equal_measure(
        positions,
        states,
        2,
        rule.config.state_dims,
        total_measure,
        grid.eps,
    )?;
    let footprint = material_footprint_radius(total_measure / particles as f32, 2);
    for row in 0..particles {
        let angle = row as f32 * 0.137;
        particle_set.closure_phase[row * 2] = angle.cos();
        particle_set.closure_phase[row * 2 + 1] = angle.sin();
        let basis = [
            angle.cos() * std::f32::consts::FRAC_1_SQRT_2 + 0.5 * angle.sin(),
            -angle.cos() * std::f32::consts::FRAC_1_SQRT_2 + 0.5 * angle.sin(),
            -0.5 * angle.sin(),
            -0.5 * angle.sin(),
        ];
        particle_set.closure_basis[row * 4..(row + 1) * 4].copy_from_slice(&basis);
        for channel in 0..particle_set.state_dims {
            particle_set.closure_mode[row * particle_set.state_dims + channel] =
                0.05 * (angle + channel as f32 * 0.17).sin();
        }
    }

    let mut adaptive = AdaptiveNpaConfig::growing_2d();
    adaptive.reference_footprint = footprint;
    adaptive.base_rule_footprint = footprint * 0.5;
    adaptive.min_footprint = footprint * 0.25;
    adaptive.max_footprint = footprint * 4.0;
    adaptive.min_leaves = particles;
    adaptive.target_leaves = particles;
    adaptive.max_leaves = particles;
    adaptive.local_residual_scale = 0.0;
    adaptive.local_residual_motion_scale = 0.0;
    adaptive.local_residual_state_scale = 0.0;
    adaptive.closure_moment_features = true;
    adaptive.closure_recurrent_mode = true;
    adaptive.proxy.enabled = false;
    adaptive.rule_graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive.perception.graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive.perception.min_bandwidth = grid.eps;
    adaptive.perception.max_bandwidth = grid.eps;
    let mut model = AdaptiveNpaModel::seeded(rule, adaptive, 313)?;
    let closure = model.closure_mode_rule.as_mut().unwrap();
    closure.weights.w1.fill(0.0);
    closure.weights.b1.fill(0.0);
    closure.weights.w2.fill(0.0);
    closure.weights.b2.fill(0.0);
    let base_dims = model.rule.config.perception_dims();
    let moment_dims = 1
        + model.config.spatial_dims * (model.config.spatial_dims + 1) / 2
        + model.rule.config.state_dims * model.config.spatial_dims;
    let phase_offset = base_dims + moment_dims + 4;
    let mode_offset = phase_offset + 2;
    let closure_context_offset = mode_offset + model.rule.config.state_dims;
    closure.weights.b1[0] = 0.7;
    closure.weights.w1[phase_offset] = 0.6;
    closure.weights.w1[phase_offset + 1] = -0.2;
    closure.weights.w1[mode_offset] = 0.4;
    closure.weights.w1[closure_context_offset] = 0.25;
    closure.weights.w2[0] = 0.03;
    closure.weights.w2[closure.config.hidden_dims] = -0.02;
    for channel in 0..closure.config.state_dims {
        closure.weights.w2[(closure.config.spatial_dims + channel) * closure.config.hidden_dims] =
            0.005 * (channel + 1) as f32;
    }
    let basis_rule = model.closure_basis_rule.as_mut().unwrap();
    basis_rule.weights.w1.fill(0.0);
    basis_rule.weights.b1.fill(0.0);
    basis_rule.weights.w2.fill(0.0);
    basis_rule.weights.b2.fill(0.0);
    basis_rule.weights.b1[0] = 0.7;
    basis_rule.weights.w1[base_dims + moment_dims] = 0.3;
    basis_rule.weights.w1[closure_context_offset] = -0.2;
    for (component, value) in [0.012, -0.012, 0.006, -0.006].into_iter().enumerate() {
        basis_rule.weights.w2[component * basis_rule.config.hidden_dims] = value;
    }
    model.validate()?;

    let initial_phase = particle_set.closure_phase.clone();
    let initial_mode = particle_set.closure_mode.clone();
    let initial_basis = particle_set.closure_basis.clone();
    let cpu = run_adaptive_rollout(
        &model,
        particle_set.clone(),
        AdaptiveRolloutConfig {
            steps: 4,
            dt: 0.25,
            update_prob: 1.0,
            seed: 317,
            bandwidth_adaptation_enabled: false,
            topology_enabled: false,
            snapshot_interval: 4,
        },
    )?;
    let mut gpu = executor.create_adaptive_state(
        &model,
        particle_set,
        &grid,
        0.25,
        burn_automata::gpu::WgpuNeighborMode::CooperativeSortedCells,
        1.0,
        317,
    )?;
    executor.step_adaptive_state_many(&mut gpu, 4, false)?;
    executor.synchronize_adaptive_particles(&mut gpu)?;

    let position_error = max_position_abs_error(&cpu.particles.positions, &gpu.particles.positions);
    let state_error = max_abs_error(&cpu.particles.states, &gpu.particles.states);
    let phase_error = max_abs_error(&cpu.particles.closure_phase, &gpu.particles.closure_phase);
    let mode_error = max_abs_error(&cpu.particles.closure_mode, &gpu.particles.closure_mode);
    let basis_error = max_abs_error(&cpu.particles.closure_basis, &gpu.particles.closure_basis);
    eprintln!(
        "recurrent closure parity: position={position_error:.7} state={state_error:.7} phase={phase_error:.7} mode={mode_error:.7} basis={basis_error:.7}"
    );
    assert!(position_error <= 4.0e-3, "position error {position_error}");
    assert!(state_error <= 4.0e-3, "state error {state_error}");
    assert!(phase_error <= 2.0e-5, "phase error {phase_error}");
    assert!(mode_error <= 2.0e-5, "mode error {mode_error}");
    assert!(basis_error <= 2.0e-5, "basis error {basis_error}");
    assert!(max_abs_error(&initial_phase, &cpu.particles.closure_phase) > 1.0e-4);
    assert!(max_abs_error(&initial_mode, &cpu.particles.closure_mode) > 1.0e-4);
    assert!(max_abs_error(&initial_basis, &cpu.particles.closure_basis) > 1.0e-4);
    Ok(())
}

#[test]
fn coarse_replacement_wgpu_step_matches_adaptive_cpu_oracle()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = wgpu_test_guard();
    let Some(executor) = new_executor_or_skip()? else {
        return Ok(());
    };
    let particles = 128;
    let (rule_config, grid) = burn_automata::NpaConfig::for_preset(AutomataPreset::Growing2d);
    let rule = NpaModel::seeded(rule_config, 223);
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        rule.config.state_dims,
        2,
        227,
        ParticleSeed::UniformCircle,
        0.08,
    );
    let total_measure = std::f32::consts::PI * 0.08_f32.powi(2);
    let mut particle_set = AdaptiveParticleSet::from_equal_measure(
        positions,
        states,
        2,
        rule.config.state_dims,
        total_measure,
        grid.eps,
    )?;
    let native_footprint = material_footprint_radius(total_measure / particles as f32, 2);
    for row in (0..particles).step_by(4) {
        particle_set.represented_measure[row] *= 4.0;
    }

    let mut adaptive = AdaptiveNpaConfig::growing_2d();
    adaptive.reference_footprint = native_footprint;
    adaptive.base_rule_footprint = native_footprint;
    adaptive.local_rule_semantics = AdaptiveLocalRuleSemantics::CoarseReplacement;
    adaptive.local_residual_scale = 0.35;
    adaptive.min_footprint = native_footprint * 0.5;
    adaptive.max_footprint = native_footprint * 3.0;
    adaptive.min_leaves = particles;
    adaptive.target_leaves = particles;
    adaptive.max_leaves = particles;
    adaptive.proxy.enabled = false;
    adaptive.rule_graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive.perception.graph_policy = AdaptiveGraphPolicy::DirectedTopK;
    adaptive.perception.max_neighbors = 128;
    adaptive.perception.min_bandwidth = grid.eps;
    adaptive.perception.max_bandwidth = grid.eps;
    let mut local = rule.clone();
    local.weights.b2[0] += 0.25;
    local.weights.b2[rule.config.spatial_dims] -= 0.15;
    let mut model = AdaptiveNpaModel::seeded(rule, adaptive, 229)?;
    model.local_residual_rule = Some(local);
    model.validate()?;

    let rollout = AdaptiveRolloutConfig {
        steps: 1,
        dt: 1.0,
        update_prob: 1.0,
        seed: 233,
        bandwidth_adaptation_enabled: false,
        topology_enabled: false,
        snapshot_interval: 1,
    };
    let cpu = run_adaptive_rollout(&model, particle_set.clone(), rollout)?;
    let mut gpu = executor.create_adaptive_state(
        &model,
        particle_set,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::CooperativeSortedCells,
        1.0,
        rollout.seed,
    )?;
    let buffers = executor.create_gaussian_buffers(particles)?;
    let bind_group = executor.create_gaussian_bind_group(&buffers.refs(), particles)?;
    executor.step_adaptive_state_many_into_gaussian_bind_group(&mut gpu, &bind_group, 1, false)?;
    let gpu = executor.read_state(&gpu.resident)?;

    let position_error = max_position_abs_error(&cpu.particles.positions, &gpu.next_positions);
    let state_error = max_abs_error(&cpu.particles.states, &gpu.next_states);
    eprintln!("coarse replacement parity: position={position_error:.7} state={state_error:.7}");
    assert!(position_error <= 3.0e-3, "position error {position_error}");
    assert!(state_error <= 3.0e-3, "state error {state_error}");
    Ok(())
}

#[test]
fn resident_wgpu_deployment_replay_preserves_snapshot_batches()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = wgpu_test_guard();
    if new_executor_or_skip()?.is_none() {
        return Ok(());
    }
    let (rule_config, grid) = burn_automata::NpaConfig::for_preset(AutomataPreset::Growing2d);
    let rule = NpaModel::seeded(rule_config.clone(), 71);
    let mut adaptive = AdaptiveNpaConfig::growing_2d();
    adaptive.min_leaves = 16;
    adaptive.initial_leaves = 16;
    adaptive.target_leaves = 64;
    adaptive.max_leaves = 64;
    adaptive.topology_start_step = 1;
    adaptive.topology_interval = 1;
    adaptive.bootstrap_end_step = 1;
    adaptive.bootstrap_events_per_interval = 16;
    adaptive.min_footprint = 0.005;
    adaptive.max_footprint = 0.2;
    adaptive.proxy.enabled = false;
    adaptive.rule_graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive.perception.graph_policy = AdaptiveGraphPolicy::RawSupport;
    let mut model = AdaptiveNpaModel::seeded(rule.clone(), adaptive, 73)?;
    model.local_residual_rule = Some(NpaModel::seeded(rule_config, 79));
    model.deployment_rule = Some(rule);
    model.validate()?;

    let config = AdaptiveMultiscaleTrainingConfig {
        on_policy_rollouts: 1,
        on_policy_rollout_steps: 4,
        on_policy_snapshot_interval: 2,
        on_policy_rows_per_snapshot: 8,
        seed_scale: 0.2,
        total_measure: std::f32::consts::PI * 0.2_f32.powi(2),
        bandwidth: 0.1,
        ..AdaptiveMultiscaleTrainingConfig::default()
    };
    let batch = adaptive_deployment_on_policy_batch_wgpu(&model, &grid, &config, 3)?;
    assert_eq!(batch.report.rollouts, 1);
    assert_eq!(batch.report.snapshots, 3);
    assert_eq!(batch.rows, 24);
    assert_eq!(batch.report.minimum_material_leaves, 16);
    assert_eq!(batch.report.maximum_material_leaves, 64);
    batch.validate(
        model.rule.config.perception_dims(),
        model.rule.config.update_dims(),
    )?;
    Ok(())
}

#[test]
fn resident_wgpu_exact_replay_preserves_snapshot_batches() -> Result<(), Box<dyn std::error::Error>>
{
    let _guard = wgpu_test_guard();
    if new_executor_or_skip()?.is_none() {
        return Ok(());
    }
    let (rule_config, grid) = burn_automata::NpaConfig::for_preset(AutomataPreset::Growing2d);
    let rule = NpaModel::seeded(rule_config.clone(), 81);
    let mut adaptive = AdaptiveNpaConfig::growing_2d();
    adaptive.min_leaves = 16;
    adaptive.initial_leaves = 16;
    adaptive.target_leaves = 64;
    adaptive.max_leaves = 64;
    adaptive.topology_start_step = 1;
    adaptive.topology_interval = 1;
    adaptive.bootstrap_end_step = 1;
    adaptive.bootstrap_events_per_interval = 16;
    adaptive.min_footprint = 0.005;
    adaptive.max_footprint = 0.2;
    adaptive.proxy.enabled = false;
    adaptive.rule_graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive.perception.graph_policy = AdaptiveGraphPolicy::RawSupport;
    let mut model = AdaptiveNpaModel::seeded(rule.clone(), adaptive, 83)?;
    model.local_residual_rule = Some(NpaModel::seeded(rule_config, 89));
    model.validate()?;

    let config = AdaptiveMultiscaleTrainingConfig {
        on_policy_replay_backend: AdaptiveReplayBackend::WgpuResident,
        on_policy_topology_control: AdaptiveTopologyControl::LocalDetailOracle,
        on_policy_rollouts: 1,
        on_policy_rollout_steps: 4,
        on_policy_snapshot_interval: 2,
        on_policy_rows_per_snapshot: 8,
        seed_scale: 0.2,
        total_measure: std::f32::consts::PI * 0.2_f32.powi(2),
        bandwidth: 0.1,
        ..AdaptiveMultiscaleTrainingConfig::default()
    };
    let batch = adaptive_multiscale_on_policy_batch(&rule, &grid, &model, &config, 5)?;
    assert_eq!(batch.report.rollouts, 1);
    assert_eq!(batch.report.snapshots, 3);
    assert_eq!(batch.rows, 24);
    assert_eq!(batch.report.minimum_material_leaves, 16);
    assert_eq!(batch.report.maximum_material_leaves, 64);
    batch.validate(rule.config.perception_dims(), rule.config.update_dims())?;
    Ok(())
}

#[test]
fn resident_wgpu_coupled_fine_replay_matches_cpu_without_stochastic_masking()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = wgpu_test_guard();
    if new_executor_or_skip()?.is_none() {
        return Ok(());
    }
    let (rule_config, grid) = burn_automata::NpaConfig::for_preset(AutomataPreset::Growing2d);
    let teacher = NpaModel::seeded(rule_config.clone(), 101);
    let mut adaptive = AdaptiveNpaConfig::growing_2d();
    adaptive.min_leaves = 8;
    adaptive.initial_leaves = 8;
    adaptive.target_leaves = 8;
    adaptive.max_leaves = 16;
    adaptive.min_footprint = 0.005;
    adaptive.max_footprint = 0.2;
    adaptive.proxy.enabled = false;
    adaptive.local_rule_semantics = burn_automata::AdaptiveLocalRuleSemantics::CompatibleResidual;
    adaptive.compatible_residual_material_features = true;
    adaptive.closure_moment_features = true;
    adaptive.reference_footprint = 0.02;
    adaptive.base_rule_footprint = 0.02;
    adaptive.rule_graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive.perception.graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive.perception.min_bandwidth = grid.eps;
    adaptive.perception.max_bandwidth = 2.0 * grid.eps;
    let mut model = AdaptiveNpaModel::seeded(teacher.clone(), adaptive, 103)?;
    let local_config = model
        .local_residual_rule
        .as_ref()
        .expect("material closure config seeds a local residual")
        .config
        .clone();
    model.local_residual_rule = Some(NpaModel::seeded(local_config, 107));
    model.validate()?;

    let base_config = AdaptiveMultiscaleTrainingConfig {
        fine_particle_count: 16,
        cut_leaf_counts: vec![8],
        on_policy_teacher: AdaptiveReplayTeacher::CoupledFine,
        on_policy_cut_steps: vec![2],
        on_policy_rollouts: 2,
        on_policy_rollout_steps: 2,
        on_policy_snapshot_interval: 1,
        on_policy_rows_per_snapshot: 8,
        update_prob: 1.0,
        seed_scale: 0.08,
        total_measure: std::f32::consts::PI * 0.08_f32.powi(2),
        bandwidth: grid.eps,
        ..AdaptiveMultiscaleTrainingConfig::default()
    };
    eprintln!("collecting compatible coupled-fine CPU reference");
    let cpu = adaptive_multiscale_on_policy_batch(&teacher, &grid, &model, &base_config, 7)?;
    eprintln!(
        "CPU reference rows={} positive_weights={} target_rms={:.6}",
        cpu.rows,
        cpu.row_weights
            .iter()
            .filter(|weight| **weight > 0.0)
            .count(),
        (cpu.target_update
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / cpu.target_update.len() as f32)
            .sqrt(),
    );
    eprintln!("collecting compatible coupled-fine WGPU replay");
    let gpu = adaptive_multiscale_on_policy_batch(
        &teacher,
        &grid,
        &model,
        &AdaptiveMultiscaleTrainingConfig {
            on_policy_replay_backend: AdaptiveReplayBackend::WgpuResident,
            ..base_config
        },
        7,
    )?;

    assert_eq!(gpu.rows, cpu.rows);
    assert_eq!(gpu.report.snapshots, cpu.report.snapshots);
    let feature_error = max_abs_error(&gpu.local_features, &cpu.local_features);
    let target_error = max_abs_error(&gpu.target_update, &cpu.target_update);
    let deployment_target_error =
        max_abs_error(&gpu.deployment_target_update, &cpu.deployment_target_update);
    let teacher_error =
        (gpu.report.mean_teacher_update_error - cpu.report.mean_teacher_update_error).abs();
    eprintln!(
        "coupled-fine replay parity: features={feature_error:.7} target={target_error:.7} deployment_target={deployment_target_error:.7} teacher_error={teacher_error:.7}",
    );
    assert!(feature_error <= 1.0e-2, "feature error {feature_error}");
    assert!(target_error <= 1.0e-2, "target error {target_error}");
    assert!(
        deployment_target_error <= 1.0e-2,
        "deployment target error {deployment_target_error}",
    );
    assert!(teacher_error <= 1.0e-3, "teacher error {teacher_error}");
    Ok(())
}

#[test]
fn resident_wgpu_recurrent_closure_replay_matches_cpu_without_stochastic_masking()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = wgpu_test_guard();
    if new_executor_or_skip()?.is_none() {
        return Ok(());
    }
    let (rule_config, grid) = burn_automata::NpaConfig::for_preset(AutomataPreset::Growing2d);
    let teacher = NpaModel::seeded(rule_config, 401);
    let mut adaptive = AdaptiveNpaConfig::growing_2d();
    adaptive.min_leaves = 13;
    adaptive.initial_leaves = 13;
    adaptive.target_leaves = 13;
    adaptive.max_leaves = 16;
    adaptive.min_footprint = 0.001;
    adaptive.max_footprint = 0.2;
    adaptive.local_residual_scale = 0.0;
    adaptive.local_residual_motion_scale = 0.0;
    adaptive.local_residual_state_scale = 0.0;
    adaptive.closure_moment_features = true;
    adaptive.closure_recurrent_mode = true;
    adaptive.proxy.enabled = false;
    adaptive.rule_graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive.perception.graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive.perception.min_bandwidth = grid.eps;
    adaptive.perception.max_bandwidth = grid.eps;
    let mut model = AdaptiveNpaModel::seeded(teacher.clone(), adaptive, 409)?;
    let local_config = model.closure_mode_rule.as_ref().unwrap().config.clone();
    model.local_residual_rule = Some(NpaModel::seeded(local_config, 419));
    model.validate()?;

    let base_config = AdaptiveMultiscaleTrainingConfig {
        fine_particle_count: 16,
        cut_leaf_counts: vec![13],
        on_policy_teacher: AdaptiveReplayTeacher::CoupledFine,
        on_policy_rollouts: 2,
        on_policy_rollout_steps: 2,
        on_policy_snapshot_interval: 1,
        on_policy_rows_per_snapshot: 13,
        update_prob: 1.0,
        seed_scale: 0.08,
        total_measure: std::f32::consts::PI * 0.08_f32.powi(2),
        bandwidth: grid.eps,
        ..AdaptiveMultiscaleTrainingConfig::default()
    };
    let cpu = adaptive_multiscale_on_policy_batch(&teacher, &grid, &model, &base_config, 11)?;
    let gpu = adaptive_multiscale_on_policy_batch(
        &teacher,
        &grid,
        &model,
        &AdaptiveMultiscaleTrainingConfig {
            on_policy_replay_backend: AdaptiveReplayBackend::WgpuResident,
            ..base_config
        },
        11,
    )?;

    assert_eq!(gpu.rows, cpu.rows);
    assert_eq!(gpu.report.snapshots, cpu.report.snapshots);
    assert_eq!(gpu.closure_mode_row_weights, cpu.closure_mode_row_weights);
    let feature_error = max_abs_error(&gpu.local_features, &cpu.local_features);
    let target_error = max_abs_error(&gpu.target_update, &cpu.target_update);
    let closure_target_error = max_abs_error(
        &gpu.closure_mode_target_update,
        &cpu.closure_mode_target_update,
    );
    let closure_basis_target_error = max_abs_error(
        &gpu.closure_basis_target_update,
        &cpu.closure_basis_target_update,
    );
    eprintln!(
        "recurrent closure replay parity: features={feature_error:.7} target={target_error:.7} closure_target={closure_target_error:.7} basis_target={closure_basis_target_error:.7}"
    );
    assert!(feature_error <= 2.0e-2, "feature error {feature_error}");
    assert!(target_error <= 2.0e-2, "target error {target_error}");
    assert!(
        closure_target_error <= 2.0e-2,
        "closure target error {closure_target_error}"
    );
    assert!(
        closure_basis_target_error <= 2.0e-2,
        "closure-basis target error {closure_basis_target_error}"
    );
    Ok(())
}

#[test]
#[ignore = "device benchmark; run explicitly with --ignored --nocapture"]
fn adaptive_wgpu_fused_throughput_matches_viewer_path() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = wgpu_test_guard();
    let Some(executor) = new_executor_or_skip()? else {
        return Ok(());
    };
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let artifact_path = std::env::var_os("BURN_AUTOMATA_ADAPTIVE_BENCH_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            workspace_root
                .join("artifacts/adaptive_npa/task_multiscale_lizard_parity/model.adaptive.bpk")
        });
    let artifact_path = if artifact_path.is_absolute() {
        artifact_path
    } else {
        workspace_root.join(artifact_path)
    };
    let artifact = burn_automata::load_adaptive_model(&artifact_path)?;
    let mut model = artifact.model;
    if std::env::var_os("BURN_AUTOMATA_ADAPTIVE_BENCH_RAW_SUPPORT").is_some() {
        model.config.perception.graph_policy = AdaptiveGraphPolicy::RawSupport;
    }
    let (_, grid) = burn_automata::NpaConfig::for_preset(AutomataPreset::Growing2d);
    let total_measure = std::f32::consts::PI * 0.2_f32.powi(2);
    let initial_count = model.config.initial_leaf_count();
    let particles = burn_automata::seed_adaptive_particles_scaled(
        &model,
        initial_count,
        42,
        ParticleSeed::UniformCircle,
        0.2,
        total_measure,
        grid.eps,
    )?;
    let mut adaptive = executor.create_adaptive_state(
        &model,
        particles,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::CooperativeSortedCells,
        0.5,
        42,
    )?;
    let buffers = executor.create_gaussian_buffers(model.config.max_leaves)?;
    let bind_group =
        executor.create_gaussian_bind_group(&buffers.refs(), model.config.max_leaves)?;

    let end_to_end_started = Instant::now();
    let end_to_end = executor.step_adaptive_state_many_into_gaussian_bind_group(
        &mut adaptive,
        &bind_group,
        256,
        true,
    )?;
    let matched_snapshot = executor.read_state(&adaptive.resident)?;
    let end_to_end_seconds = end_to_end_started.elapsed().as_secs_f64();
    let mut topology_ms = end_to_end
        .topology_updates
        .iter()
        .map(|update| update.elapsed_ms)
        .collect::<Vec<_>>();
    topology_ms.sort_by(f64::total_cmp);
    let topology_total_ms = topology_ms.iter().sum::<f64>();
    let topology_median_ms = topology_ms
        .get(topology_ms.len() / 2)
        .copied()
        .unwrap_or_default();
    let topology_max_ms = topology_ms.last().copied().unwrap_or_default();
    let topology_timeline = end_to_end
        .topology_updates
        .iter()
        .map(|update| format!("{}:{:.1}", update.step, update.elapsed_ms))
        .collect::<Vec<_>>()
        .join(",");

    executor.step_adaptive_state_many_into_gaussian_bind_group(
        &mut adaptive,
        &bind_group,
        128,
        false,
    )?;
    const TIMED_STEPS: usize = 2_048;
    let steady_started = Instant::now();
    let mut steady_particle_steps = 0_usize;
    let mut steady = None;
    for _ in 0..TIMED_STEPS {
        let report = executor.step_adaptive_state_many_into_gaussian_bind_group(
            &mut adaptive,
            &bind_group,
            1,
            false,
        )?;
        steady_particle_steps += report.particle_steps;
        steady = Some(report);
    }
    let steady = steady.expect("timed adaptive benchmark has at least one step");
    let _ = executor.read_state(&adaptive.resident)?;
    let steady_seconds = steady_started.elapsed().as_secs_f64();

    let visible_count = steady.resident_particle_count;
    let dynamics_count = steady.dynamics_particle_count;
    assert_eq!(matched_snapshot.next_positions.len(), dynamics_count);
    let same_positions = matched_snapshot.next_positions;
    let same_states = matched_snapshot.next_states;
    let mut dynamics_control = executor.create_state_with_neighbor_mode_and_update_prob(
        &model.rule,
        &same_positions,
        &same_states,
        1,
        dynamics_count,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::CooperativeSortedCells,
        0.5,
        42,
    )?;
    let dynamics_buffers = executor.create_gaussian_buffers(dynamics_count)?;
    let dynamics_bind_group =
        executor.create_gaussian_bind_group(&dynamics_buffers.refs(), dynamics_count)?;
    executor.step_state_many_into_gaussian_bind_group(
        &mut dynamics_control,
        &dynamics_bind_group,
        128,
    )?;
    let dynamics_started = Instant::now();
    for _ in 0..TIMED_STEPS {
        executor.step_state_many_into_gaussian_bind_group(
            &mut dynamics_control,
            &dynamics_bind_group,
            1,
        )?;
    }
    let _ = executor.read_state(&dynamics_control)?;
    let dynamics_seconds = dynamics_started.elapsed().as_secs_f64();

    let (visible_positions, visible_states) = seed_particles_scaled(
        1,
        visible_count,
        model.rule.config.state_dims,
        2,
        42,
        ParticleSeed::UniformCircle,
        0.2,
    );
    let mut visible_control = executor.create_state_with_neighbor_mode_and_update_prob(
        &model.rule,
        &visible_positions,
        &visible_states,
        1,
        visible_count,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::CooperativeSortedCells,
        0.5,
        42,
    )?;
    let visible_buffers = executor.create_gaussian_buffers(visible_count)?;
    let visible_bind_group =
        executor.create_gaussian_bind_group(&visible_buffers.refs(), visible_count)?;
    executor.step_state_many_into_gaussian_bind_group(
        &mut visible_control,
        &visible_bind_group,
        128,
    )?;
    let visible_started = Instant::now();
    for _ in 0..TIMED_STEPS {
        executor.step_state_many_into_gaussian_bind_group(
            &mut visible_control,
            &visible_bind_group,
            1,
        )?;
    }
    let _ = executor.read_state(&visible_control)?;
    let visible_seconds = visible_started.elapsed().as_secs_f64();

    let regular_count = model.config.max_leaves;
    let (regular_positions, regular_states) = seed_particles_scaled(
        1,
        regular_count,
        model.rule.config.state_dims,
        2,
        42,
        ParticleSeed::UniformCircle,
        0.2,
    );
    let mut regular = executor.create_state_with_neighbor_mode_and_update_prob(
        &model.rule,
        &regular_positions,
        &regular_states,
        1,
        regular_count,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::CooperativeSortedCells,
        0.5,
        42,
    )?;
    let regular_buffers = executor.create_gaussian_buffers(regular_count)?;
    let regular_bind_group =
        executor.create_gaussian_bind_group(&regular_buffers.refs(), regular_count)?;
    executor.step_state_many_into_gaussian_bind_group(&mut regular, &regular_bind_group, 128)?;
    let regular_started = Instant::now();
    for _ in 0..TIMED_STEPS {
        executor.step_state_many_into_gaussian_bind_group(&mut regular, &regular_bind_group, 1)?;
    }
    let _ = executor.read_state(&regular)?;
    let regular_seconds = regular_started.elapsed().as_secs_f64();

    let end_to_end_rate = end_to_end.particle_steps as f64 / end_to_end_seconds;
    let steady_rate = steady_particle_steps as f64 / steady_seconds;
    let dynamics_rate = (dynamics_count * TIMED_STEPS) as f64 / dynamics_seconds;
    let visible_rate = (visible_count * TIMED_STEPS) as f64 / visible_seconds;
    let regular_rate = (regular_count * TIMED_STEPS) as f64 / regular_seconds;
    let restriction_export_ms =
        (steady_seconds - dynamics_seconds).max(0.0) * 1_000.0 / TIMED_STEPS as f64;
    eprintln!(
        "adaptive WGPU viewer cadence: graph={:?}; end-to-end={:.3} ms/step ({:.2}M particle-steps/s, {}->{} visible/{} dynamics, {} topology passes totaling {:.1} ms, median={:.1} ms, max={:.1} ms); steady={:.3} ms/step ({:.2}M particle-steps/s); restriction/export={:.3} ms/step; regular-dynamics={:.3} ms/step ({:.2}M particle-steps/s); regular-visible={:.3} ms/step ({:.2}M particle-steps/s); regular-4096={:.3} ms/step ({:.2}M particle-steps/s); wall ratio dynamics={:.3}x, visible={:.3}x, 4096={:.3}x",
        model.config.perception.graph_policy,
        end_to_end_seconds * 1_000.0 / 256.0,
        end_to_end_rate / 1.0e6,
        initial_count,
        end_to_end.resident_particle_count,
        end_to_end.dynamics_particle_count,
        end_to_end.topology_updates.len(),
        topology_total_ms,
        topology_median_ms,
        topology_max_ms,
        steady_seconds * 1_000.0 / TIMED_STEPS as f64,
        steady_rate / 1.0e6,
        restriction_export_ms,
        dynamics_seconds * 1_000.0 / TIMED_STEPS as f64,
        dynamics_rate / 1.0e6,
        visible_seconds * 1_000.0 / TIMED_STEPS as f64,
        visible_rate / 1.0e6,
        regular_seconds * 1_000.0 / TIMED_STEPS as f64,
        regular_rate / 1.0e6,
        steady_seconds / dynamics_seconds,
        steady_seconds / visible_seconds,
        steady_seconds / regular_seconds,
    );
    eprintln!("adaptive topology timeline step:ms=[{topology_timeline}]");
    assert!(
        restriction_export_ms <= 0.25,
        "adaptive restriction/export overhead regressed: {restriction_export_ms:.3} ms/step",
    );
    assert!(
        steady_seconds <= regular_seconds * 1.05,
        "budgeted adaptive viewer path regressed against regular 4096: {:.3}x",
        steady_seconds / regular_seconds,
    );
    Ok(())
}

#[test]
#[ignore = "artifact-level device quality regression; run explicitly with --ignored --nocapture"]
fn adaptive_wgpu_lod_artifact_preserves_worst_seed_quality()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = wgpu_test_guard();
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let resolve = |path: std::path::PathBuf| {
        if path.is_absolute() {
            path
        } else {
            workspace_root.join(path)
        }
    };
    let model_path = std::env::var_os("BURN_AUTOMATA_ADAPTIVE_BENCH_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            workspace_root
                .join("artifacts/adaptive_npa/lod_progressive_mixed_3070/model.adaptive.bpk")
        });
    let model_path = resolve(model_path);
    let config_path = std::env::var_os("BURN_AUTOMATA_ADAPTIVE_QUALITY_CONFIG")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            workspace_root.join(
                "configs/verified/2d/adaptive/evaluation/task_lod_lizard_smoke_3070_2d_wgpu.toml",
            )
        });
    let config_path = resolve(config_path);
    if !model_path.is_file() || !config_path.is_file() {
        eprintln!(
            "skipping adaptive LoD artifact regression; model={} config={}",
            model_path.display(),
            config_path.display(),
        );
        return Ok(());
    }
    let mut config: AdaptiveExperimentConfig =
        toml::from_str(&std::fs::read_to_string(config_path)?)?;
    config.task_quality.target_image = resolve(config.task_quality.target_image);
    config.task_quality.reference_model = config.task_quality.reference_model.map(&resolve);
    let artifact = load_adaptive_model(model_path)?;
    let report = evaluate_adaptive_task_quality_validation(&artifact.model, &config, &[114, 115])?;
    let failures = validate_adaptive_task_quality_validation_gates(config.gates, Some(&report));
    assert!(
        failures.is_empty(),
        "adaptive LoD worst-seed quality regressed:\n- {}",
        failures.join("\n- "),
    );
    assert_eq!(report.rows.len(), 2);
    assert!(report.rows.iter().any(|row| row.seed == 114));
    eprintln!(
        "adaptive LoD artifact gate: adaptive={:.3} dB regular-gap={:+.3}/{:+.3} dB topology={:+.3}/{:+.3} dB",
        report.mean_adaptive_target_composited_psnr_db,
        report.mean_adaptive_over_regular_base_psnr_gain_db,
        report.worst_adaptive_over_regular_base_psnr_gain_db,
        report.mean_adaptive_over_budget_fixed_psnr_gain_db,
        report.worst_adaptive_over_budget_fixed_psnr_gain_db,
    );
    Ok(())
}

#[test]
fn represented_measure_wgpu_gaussians_preserve_continuous_material_scale()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = wgpu_test_guard();
    let Some(executor) = new_executor_or_skip()? else {
        return Ok(());
    };
    let particles = 64;
    let (config, grid) = burn_automata::NpaConfig::for_preset(AutomataPreset::Growing2d);
    let model = NpaModel {
        weights: NpaWeights::zeros(&config),
        config,
    };
    let (positions, states) = seed_particles_scaled(
        1,
        particles,
        model.config.state_dims,
        2,
        79,
        ParticleSeed::UniformCircle,
        0.2,
    );
    let measures = (0..particles)
        .map(|index| {
            let radius = 0.0025 + 0.00008 * index as f32;
            std::f32::consts::PI * radius.powi(2)
        })
        .collect::<Vec<_>>();
    let covariance = measures
        .iter()
        .enumerate()
        .map(|(index, measure)| {
            let radius = material_footprint_radius(*measure, 2);
            let variance = (0.5 * radius).powi(2);
            if index.is_multiple_of(2) {
                [
                    16.0 * variance,
                    0.75 * variance,
                    0.0,
                    0.75 * variance,
                    0.25 * variance,
                    0.0,
                    0.0,
                    0.0,
                    variance,
                ]
            } else {
                [variance, 0.0, 0.0, 0.0, variance, 0.0, 0.0, 0.0, variance]
            }
        })
        .collect::<Vec<_>>();
    let display_scale = 3.25;
    let render_from = measures
        .iter()
        .map(|measure| material_footprint_radius(*measure, 2) * display_scale)
        .collect::<Vec<_>>();
    let render_target_footprint = measures
        .iter()
        .map(|measure| material_footprint_radius(*measure, 2))
        .collect::<Vec<_>>();
    let mut state = executor.create_material_state_with_neighbor_mode_and_update_prob(
        &model,
        &positions,
        &states,
        1,
        particles,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::CooperativeSortedCells,
        1.0,
        5,
        WgpuMaterialStateInit {
            represented_measure: &measures,
            particle_ids: None,
            update_masks: None,
            bandwidth: &vec![grid.eps; particles],
            support_bins: None,
            covariance: &covariance,
            state_jacobian: &vec![
                0.0;
                particles * model.config.state_dims * model.config.spatial_dims
            ],
            closure_mode: None,
            closure_basis: None,
            closure_phase: None,
            render_from_scale: &render_from,
            render_target_footprint: &render_target_footprint,
            display_scale_per_footprint: display_scale,
            render_transition_steps: 0,
        },
    )?;
    let buffers = executor.create_gaussian_buffers(particles)?;
    executor.step_state_into_gaussians(&mut state, &buffers.refs())?;
    let rendered = executor.read_gaussian_buffers(&buffers)?;
    for index in 0..particles {
        let expected = adaptive_isotropic_gaussian_geometry(
            measures[index],
            render_target_footprint[index],
            2,
        )?;
        let base = index * 4;
        for axis in 0..3 {
            let expected_scale = expected.scale[axis] * display_scale;
            let actual = rendered.scale_opacity[base + axis];
            assert!(
                (actual - expected_scale).abs() <= 2.0e-6,
                "particle {index} axis {axis}: expected {expected_scale}, got {actual}"
            );
        }
        for component in 0..4 {
            let actual = rendered.rotation[base + component];
            assert!(
                (actual - expected.rotation[component]).abs() <= 2.0e-6,
                "particle {index} rotation component {component}: expected {}, got {actual}",
                expected.rotation[component],
            );
        }
        assert!((rendered.scale_opacity[base + 3] - expected.opacity).abs() <= 2.0e-6);
    }
    for pair in rendered
        .scale_opacity
        .chunks_exact(4)
        .collect::<Vec<_>>()
        .windows(2)
    {
        assert!(
            pair[1][0] > pair[0][0],
            "continuous material radii were quantized or reordered"
        );
    }
    Ok(())
}

#[test]
fn adaptive_wgpu_resident_rollout_bootstraps_coarse_material_without_scale_snap()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = wgpu_test_guard();
    let Some(executor) = new_executor_or_skip()? else {
        return Ok(());
    };
    let (config, grid) = burn_automata::NpaConfig::for_preset(AutomataPreset::Growing2d);
    let rule = NpaModel {
        weights: NpaWeights::zeros(&config),
        config,
    };
    let mut adaptive = AdaptiveNpaConfig::growing_2d();
    adaptive.min_leaves = 4;
    adaptive.initial_leaves = 4;
    adaptive.target_leaves = 64;
    adaptive.max_leaves = 64;
    adaptive.reference_footprint = 0.05;
    adaptive.base_rule_footprint = 0.05;
    adaptive.min_footprint = 0.025;
    adaptive.max_footprint = 0.2;
    adaptive.topology_interval = 1;
    adaptive.topology_start_step = 1;
    adaptive.max_events_per_interval = 1;
    adaptive.bootstrap_end_step = 2;
    adaptive.bootstrap_events_per_interval = 64;
    adaptive.render_transition_steps = 8;
    adaptive.cooldown_steps = 0;
    adaptive.proxy.enabled = false;
    adaptive.perception.graph_policy = AdaptiveGraphPolicy::RawSupport;
    adaptive.rule_graph_policy = AdaptiveGraphPolicy::RawSupport;
    let model = AdaptiveNpaModel::seeded(rule, adaptive, 9)?;
    let (positions, states) = seed_particles_scaled(
        1,
        4,
        model.rule.config.state_dims,
        2,
        7,
        ParticleSeed::UniformCircle,
        0.2,
    );
    let particles = AdaptiveParticleSet::from_equal_measure(
        positions,
        states,
        2,
        model.rule.config.state_dims,
        std::f32::consts::PI * 0.2_f32.powi(2),
        grid.eps,
    )?;
    let mut state = executor.create_adaptive_state(
        &model,
        particles,
        &grid,
        1.0,
        burn_automata::gpu::WgpuNeighborMode::CooperativeSortedCells,
        1.0,
        11,
    )?;
    let buffers = executor.create_gaussian_buffers(64)?;
    let bind_group = executor.create_gaussian_bind_group(&buffers.refs(), 64)?;
    executor.write_adaptive_state_into_gaussian_bind_group(&mut state, &bind_group)?;
    let initial = executor.read_gaussian_buffers(&buffers)?;
    for (index, expected) in state.particles.positions.iter().enumerate() {
        let base = index * 4;
        assert!(
            (initial.position_visibility[base] - expected[0]).abs() <= 2.0e-6
                && (initial.position_visibility[base + 1] - expected[1]).abs() <= 2.0e-6,
            "initial adaptive Gaussian row {index} was not restricted to visible material",
        );
    }
    let report = executor.step_adaptive_state_many_into_gaussian_bind_group(
        &mut state,
        &bind_group,
        2,
        true,
    )?;
    assert_eq!(report.resident_particle_count, 64);
    assert_eq!(report.topology_updates.len(), 2);
    assert_eq!(report.topology_updates[0].split_events, 4);
    assert_eq!(report.topology_updates[1].split_events, 16);
    let gaussians = executor.read_gaussian_buffers(&buffers)?;
    let displayed = gaussians
        .scale_opacity
        .chunks_exact(4)
        .take(64)
        .map(|scale| scale[0])
        .collect::<Vec<_>>();
    let spread = displayed.iter().copied().fold(0.0_f32, f32::max)
        - displayed.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread <= 1.0e-5,
        "split children snapped to inconsistent scales"
    );
    let target_fine_footprint =
        material_footprint_radius(std::f32::consts::PI * 0.2_f32.powi(2) / 64.0, 2);
    let target_fine_scale =
        target_fine_footprint * burn_automata::adaptive_display_scale_per_footprint(&model);
    assert!(
        displayed
            .iter()
            .all(|scale| *scale > target_fine_scale * 3.0),
        "new split leaves snapped directly to fine scale"
    );

    executor.step_adaptive_state_many_into_gaussian_bind_group(
        &mut state,
        &bind_group,
        4,
        false,
    )?;
    let midpoint = executor.read_gaussian_buffers(&buffers)?;
    let midpoint_scale = midpoint.scale_opacity[0];
    assert!(
        midpoint_scale > target_fine_scale && midpoint_scale < displayed[0],
        "display scale should transition continuously, got coarse={} midpoint={midpoint_scale} target={target_fine_scale}",
        displayed[0],
    );

    executor.step_adaptive_state_many_into_gaussian_bind_group(
        &mut state,
        &bind_group,
        4,
        false,
    )?;
    let settled = executor.read_gaussian_buffers(&buffers)?;
    assert!(
        (settled.scale_opacity[0] - target_fine_scale).abs() <= 2.0e-5,
        "display scale did not settle to the physical fine scale: got {}, target {target_fine_scale}",
        settled.scale_opacity[0],
    );
    Ok(())
}
