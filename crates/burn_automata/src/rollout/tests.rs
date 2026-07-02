use super::*;

fn active_seed_count(states: &[f32], state_dims: usize) -> usize {
    states
        .chunks_exact(state_dims)
        .filter(|state| state[3] > -1.0)
        .count()
}

#[test]
fn growth_3d_seed_active_count_uses_minimum_3d_nucleus_for_small_clouds() {
    assert_eq!(growth_3d_active_seed_count(0), 0);
    assert_eq!(growth_3d_active_seed_count(4), 4);
    assert_eq!(
        growth_3d_active_seed_count(64),
        GROWTH_3D_MIN_ACTIVE_SEED_COUNT
    );
    assert_eq!(
        growth_3d_active_seed_count(128),
        GROWTH_3D_MIN_ACTIVE_SEED_COUNT
    );

    let proportional_count =
        (1024.0_f32 * GROWTH_3D_ACTIVE_CORE_RADIUS_RATIO.powi(3)).round() as usize;
    assert!(proportional_count > GROWTH_3D_MIN_ACTIVE_SEED_COUNT);
    assert_eq!(growth_3d_active_seed_count(1024), proportional_count);
}

#[test]
fn growth_3d_seed_active_count_is_seed_stable() {
    let state_dims = 12;
    let particle_count = 1024;
    let expected = growth_3d_active_seed_count(particle_count);
    for seed in [1, 42, 99, 0x005a_173d, 0x0051_a73d, 0xffff_ffff] {
        let (_positions, states) = seed_particles_scaled(
            1,
            particle_count,
            state_dims,
            3,
            seed,
            ParticleSeed::TeapotGrowth3d,
            0.72,
        );
        assert_eq!(active_seed_count(&states, state_dims), expected);
    }
}

#[test]
fn growth_3d_seed_positions_vary_by_seed_with_stable_radius_distribution() {
    let particle_count = 128;
    let (positions_a, states_a) = seed_particles_scaled(
        1,
        particle_count,
        12,
        3,
        42,
        ParticleSeed::TeapotGrowth3d,
        0.72,
    );
    let (positions_b, states_b) = seed_particles_scaled(
        1,
        particle_count,
        12,
        3,
        0x005a_173d,
        ParticleSeed::TeapotGrowth3d,
        0.72,
    );
    assert_ne!(
        positions_a, positions_b,
        "held-out 3D growth seeds should not replay the same stratified cloud"
    );
    assert_ne!(
        states_a, states_b,
        "seed-frame coordinate state should follow the seed-specific positions"
    );

    let mut radii_a = positions_a
        .iter()
        .map(|position| (position[0].powi(2) + position[1].powi(2) + position[2].powi(2)).sqrt())
        .collect::<Vec<_>>();
    let mut radii_b = positions_b
        .iter()
        .map(|position| (position[0].powi(2) + position[1].powi(2) + position[2].powi(2)).sqrt())
        .collect::<Vec<_>>();
    radii_a.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    radii_b.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    for (lhs, rhs) in radii_a.iter().zip(radii_b.iter()) {
        assert!(
            (lhs - rhs).abs() < 1.0e-6,
            "seed variation should rotate/jitter the stratified cloud without changing its radial curriculum"
        );
    }
}

#[test]
fn growth_3d_seed_positions_are_stratified_inside_expected_radii() {
    let particle_count = 512;
    let scale = 0.72;
    let active_count = growth_3d_active_seed_count(particle_count);
    let active_radius = growth_3d_active_core_radius(scale);
    let seed_radius = growth_3d_seed_radius(scale);
    let domain_radius = growth_3d_domain_radius(scale);
    let (positions, states) = seed_particles_scaled(
        1,
        particle_count,
        12,
        3,
        0x5eed,
        ParticleSeed::TorusGrowth3d,
        scale,
    );

    for (idx, position) in positions.iter().enumerate() {
        let radius =
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt();
        assert!(radius <= seed_radius + 1.0e-5);
        let state_base = idx * 12;
        for axis in 0..3 {
            assert!(
                (states[state_base + axis] - position[axis] / domain_radius).abs() < 1.0e-6,
                "growth seed state channels 0..2 should store normalized seed-frame coordinates"
            );
        }
        if idx < active_count {
            assert!(radius <= active_radius + 1.0e-5);
            assert_eq!(states[idx * 12 + 3], GROWTH_3D_ACTIVE_OPACITY_LOGIT);
            assert_eq!(
                states[idx * 12 + GROWTH_3D_RENDER_OPACITY_CHANNEL],
                GROWTH_3D_ACTIVE_OPACITY_LOGIT
            );
        } else {
            assert!(radius >= active_radius - 1.0e-5);
            assert_eq!(states[idx * 12 + 3], GROWTH_3D_INACTIVE_OPACITY_LOGIT);
            assert_eq!(
                states[idx * 12 + GROWTH_3D_RENDER_OPACITY_CHANNEL],
                GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT
            );
        }
    }
}

#[test]
fn growth_3d_substrate_seed_keeps_sparse_active_core_in_dormant_domain() {
    for particle_count in [64, 512] {
        let scale = 0.72;
        let active_count = growth_3d_active_seed_count(particle_count);
        let active_radius = growth_3d_active_core_radius(scale);
        let seed_radius = growth_3d_seed_radius(scale);
        let domain_radius = growth_3d_domain_radius(scale);
        let inactive_count = particle_count.saturating_sub(active_count);
        let ray_count = growth_3d_substrate_ray_count(inactive_count, active_count, scale)
            .min(inactive_count.max(1));
        let kernel_radius = HashGridConfig::growing_3dgs().eps;
        let (positions, states) = seed_particles_scaled(
            1,
            particle_count,
            12,
            3,
            0x5eed,
            ParticleSeed::TorusSubstrateGrowth3d,
            scale,
        );

        let mut max_radius = 0.0_f32;
        let mut inactive_local_shell = 0usize;
        let mut inactive_beyond_seed = 0usize;
        let mut ray_radii = vec![Vec::<f32>::new(); ray_count];
        let mut active_positions = Vec::new();
        let mut first_ray_positions = vec![None::<[f32; 4]>; ray_count];
        for (idx, position) in positions.iter().enumerate() {
            let radius =
                (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                    .sqrt();
            max_radius = max_radius.max(radius);
            assert!(radius <= domain_radius + 1.0e-5);
            let state_base = idx * 12;
            for axis in 0..3 {
                assert!(
                    (states[state_base + axis] - position[axis] / domain_radius).abs() < 1.0e-6,
                    "substrate seed state channels 0..2 should store normalized seed-frame coordinates"
                );
            }
            if idx < active_count {
                active_positions.push(*position);
                assert!(radius <= active_radius + 1.0e-5);
                assert_eq!(states[idx * 12 + 3], GROWTH_3D_ACTIVE_OPACITY_LOGIT);
                assert_eq!(
                    states[idx * 12 + GROWTH_3D_RENDER_OPACITY_CHANNEL],
                    GROWTH_3D_ACTIVE_OPACITY_LOGIT
                );
            } else {
                assert!(radius >= active_radius - 1.0e-5);
                if radius <= seed_radius + 1.0e-5 {
                    inactive_local_shell += 1;
                }
                if radius > seed_radius {
                    inactive_beyond_seed += 1;
                }
                let inactive_idx = idx - active_count;
                let ray = inactive_idx % ray_count;
                if first_ray_positions[ray].is_none() {
                    first_ray_positions[ray] = Some(*position);
                }
                ray_radii[ray].push(radius);
                assert_eq!(
                    states[idx * 12 + 3],
                    GROWTH_3D_SUBSTRATE_INACTIVE_OPACITY_LOGIT
                );
                assert_eq!(
                    states[idx * 12 + GROWTH_3D_RENDER_OPACITY_CHANNEL],
                    GROWTH_3D_MATERIAL_INACTIVE_OPACITY_LOGIT
                );
            }
        }

        assert!(max_radius > domain_radius * 0.95);
        assert!(
            inactive_local_shell > active_count,
            "substrate seed should include a dormant local shell adjacent to the active core"
        );
        assert!(inactive_beyond_seed > particle_count / 2);
        for radii in &mut ray_radii {
            radii.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap());
            for pair in radii.windows(2) {
                assert!(
                    pair[1] - pair[0] < kernel_radius,
                    "substrate radial rays should stay inside the kernel support"
                );
            }
        }
        for first_position in first_ray_positions.into_iter().flatten() {
            let nearest_active = active_positions
                .iter()
                .map(|active| {
                    let dx = first_position[0] - active[0];
                    let dy = first_position[1] - active[1];
                    let dz = first_position[2] - active[2];
                    (dx * dx + dy * dy + dz * dz).sqrt()
                })
                .fold(f32::MAX, f32::min);
            assert!(
                nearest_active < kernel_radius,
                "each substrate ray should begin within kernel support of the active core"
            );
        }
    }
}

#[test]
fn growth_3d_local_substrate_seed_keeps_topology_without_coordinate_state() {
    let particle_count = 128;
    let scale = 0.72;
    let active_count = growth_3d_active_seed_count(particle_count);
    let inactive_count = particle_count.saturating_sub(active_count);
    let ray_count = growth_3d_substrate_ray_count(inactive_count, active_count, scale)
        .min(inactive_count.max(1));
    let kernel_radius = HashGridConfig::growing_3dgs().eps;
    let (positions, states) = seed_particles_scaled(
        1,
        particle_count,
        12,
        3,
        0x5eed,
        ParticleSeed::TorusLocalSubstrateGrowth3d,
        scale,
    );

    let mut active_positions = Vec::new();
    let mut first_ray_positions = vec![None::<[f32; 4]>; ray_count];
    let mut max_radius = 0.0_f32;
    for (idx, position) in positions.iter().enumerate() {
        let state_base = idx * 12;
        assert_eq!(
            states[state_base], 0.0,
            "local substrate seed must not write x scaffold state"
        );
        assert_eq!(
            states[state_base + 1],
            0.0,
            "local substrate seed must not write y scaffold state"
        );
        assert_eq!(
            states[state_base + 2],
            0.0,
            "local substrate seed must not write z scaffold state"
        );
        let radius =
            (position[0] * position[0] + position[1] * position[1] + position[2] * position[2])
                .sqrt();
        max_radius = max_radius.max(radius);
        if idx < active_count {
            active_positions.push(*position);
            assert_eq!(states[state_base + 3], GROWTH_3D_ACTIVE_OPACITY_LOGIT);
        } else {
            assert_eq!(
                states[state_base + 3],
                GROWTH_3D_SUBSTRATE_INACTIVE_OPACITY_LOGIT
            );
            let inactive_idx = idx - active_count;
            let ray = inactive_idx % ray_count;
            if first_ray_positions[ray].is_none() {
                first_ray_positions[ray] = Some(*position);
            }
        }
    }

    assert!(max_radius > growth_3d_domain_radius(scale) * 0.95);
    for first_position in first_ray_positions.into_iter().flatten() {
        let nearest_active = active_positions
            .iter()
            .map(|active| {
                let dx = first_position[0] - active[0];
                let dy = first_position[1] - active[1];
                let dz = first_position[2] - active[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .fold(f32::MAX, f32::min);
        assert!(
            nearest_active < kernel_radius,
            "local substrate seed should keep first dormant shell within kernel support"
        );
    }
}
