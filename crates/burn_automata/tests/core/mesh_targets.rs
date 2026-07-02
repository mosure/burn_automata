use super::*;

#[test]
fn uv_torus_3d_seed_places_particles_on_colored_torus() {
    let particles = 256;
    let state_dims = 8;
    let scale = 0.72;
    let (positions, states) = seed_particles_scaled(
        2,
        particles,
        state_dims,
        3,
        29,
        ParticleSeed::UvTorus3d,
        scale,
    );
    let major = scale * UV_TORUS_INITIAL_SCALE;
    let minor = major * UV_TORUS_MINOR_RATIO;
    let mut max_target_error = 0.0_f32;
    let mut max_residual_error = 0.0_f32;
    let mut max_color_error = 0.0_f32;

    assert_eq!(positions.len(), particles * 2);
    assert_eq!(states.len(), particles * 2 * state_dims);
    for (idx, position) in positions.iter().enumerate() {
        let radial = (position[0] * position[0] + position[1] * position[1]).sqrt();
        let torus_radius = ((radial - major).powi(2) + position[2].powi(2)).sqrt();
        assert!(
            (torus_radius - minor).abs() < 2.0e-5,
            "particle {idx}: torus radius {torus_radius}, expected {minor}"
        );

        let state_base = idx * state_dims;
        let target = uv_torus_sample(idx % particles, particles, scale).position;
        let reconstructed = [
            position[0] + states[state_base],
            position[1] + states[state_base + 1],
            position[2] + states[state_base + 2],
        ];
        let target_error = ((reconstructed[0] - target[0]).powi(2)
            + (reconstructed[1] - target[1]).powi(2)
            + (reconstructed[2] - target[2]).powi(2))
        .sqrt();
        max_target_error = max_target_error.max(target_error);
        let residual_error = ((states[state_base] - (target[0] - position[0])).powi(2)
            + (states[state_base + 1] - (target[1] - position[1])).powi(2)
            + (states[state_base + 2] - (target[2] - position[2])).powi(2))
        .sqrt();
        max_residual_error = max_residual_error.max(residual_error);
        assert!((states[state_base + 3] - UV_TORUS_INITIAL_OPACITY_LOGIT).abs() < 1.0e-6);
        let actual_rgb = uv_torus_tail_state_to_rgb([
            states[state_base + state_dims - 3],
            states[state_base + state_dims - 2],
            states[state_base + state_dims - 1],
        ]);
        let expected_rgb = uv_torus_position_color(target, scale);
        let color_error = ((actual_rgb[0] - expected_rgb[0]).powi(2)
            + (actual_rgb[1] - expected_rgb[1]).powi(2)
            + (actual_rgb[2] - expected_rgb[2]).powi(2))
        .sqrt();
        max_color_error = max_color_error.max(color_error);
    }

    assert!(max_target_error <= 2.0e-5);
    assert!(max_residual_error <= 2.0e-5);
    assert!(max_color_error <= 1.0e-6);
}
#[test]
fn uv_torus_sampler_uses_independent_ring_and_tube_axes() {
    let particles = 256;
    let scale = 0.72;
    let first = uv_torus_sample(0, particles, scale);
    let same_tube_next_ring = uv_torus_sample(1, particles, scale);
    let next_tube_first_ring = uv_torus_sample(16, particles, scale);

    assert!(same_tube_next_ring.u > first.u);
    assert!((same_tube_next_ring.v - first.v).abs() <= f32::EPSILON);
    assert!((next_tube_first_ring.u - first.u).abs() <= f32::EPSILON);
    assert!(next_tube_first_ring.v > first.v);
}
#[test]
fn uv_torus_continuous_samples_use_implicit_surface_and_volume() {
    let scale = 0.72;
    let mut rng = StdRng::seed_from_u64(0x70_75);
    let mut max_surface_error = 0.0_f32;
    let mut max_projected_volume_error = 0.0_f32;
    let mut accumulated_surface_delta = 0.0_f32;
    let mut previous_surface = uv_torus_continuous_surface_position(&mut rng, scale);

    for _ in 0..128 {
        let surface = uv_torus_continuous_surface_position(&mut rng, scale);
        max_surface_error = max_surface_error.max(uv_torus_surface_error(surface, scale));
        accumulated_surface_delta += ((surface[0] - previous_surface[0]).powi(2)
            + (surface[1] - previous_surface[1]).powi(2)
            + (surface[2] - previous_surface[2]).powi(2))
        .sqrt();
        previous_surface = surface;

        let volume = uv_torus_continuous_volume_position(&mut rng, scale);
        let projected = uv_torus_project_position(volume, scale);
        max_projected_volume_error =
            max_projected_volume_error.max(uv_torus_surface_error(projected, scale));
    }

    assert!(max_surface_error <= 2.0e-5);
    assert!(max_projected_volume_error <= 2.0e-5);
    assert!(
        accumulated_surface_delta > scale,
        "continuous surface sampler did not cover a meaningful torus arc"
    );
}
#[test]
fn uv_torus_mesh_target_keeps_inner_and_outer_curvature_oriented() {
    let scale = 0.72;
    let minor = scale * UV_TORUS_MINOR_RATIO;
    let target = TriangleMeshTarget::torus(scale, minor, 96, 64).unwrap();

    let outer = target.project([scale + minor + 0.08, 0.0, 0.0]);
    let inner_hole = target.project([scale - minor - 0.08, 0.0, 0.0]);
    let inside_solid = target.project([scale + 0.5 * minor, 0.0, 0.0]);

    assert!(outer.signed_distance > 0.07);
    assert!(inner_hole.signed_distance > 0.07);
    assert!(inside_solid.signed_distance < -0.25);
    assert!(dot3(outer.normal, [1.0, 0.0, 0.0]) > 0.99);
    assert!(dot3(inner_hole.normal, [-1.0, 0.0, 0.0]) > 0.99);
    assert!(
        dot3(outer.normal, inner_hole.normal) < -0.99,
        "inner and outer tube normals should point in opposite directions"
    );
    assert!(uv_torus_surface_error(outer.closest, scale) < 2.0e-3);
    assert!(uv_torus_surface_error(inner_hole.closest, scale) < 2.0e-3);
    assert!((outer.signed_distance - uv_torus_signed_distance(outer.query, scale)).abs() < 2.0e-3);
    assert!(
        (inner_hole.signed_distance - uv_torus_signed_distance(inner_hole.query, scale)).abs()
            < 2.0e-3
    );
}
#[test]
fn mesh_surface_samples_cover_torus_surface_without_face_prefix_bias() {
    let scale = 0.72;
    let minor = scale * UV_TORUS_MINOR_RATIO;
    let target = TriangleMeshTarget::torus(scale, minor, 96, 64).unwrap();
    let ring_bins = 24usize;
    let tube_bins = 16usize;
    let mut covered_rings = HashSet::new();
    let mut covered_tubes = HashSet::new();

    for sample_idx in 0..512 {
        let sample = target.surface_sample(sample_idx);
        let theta = sample.position[1].atan2(sample.position[0]);
        let radial = (sample.position[0] * sample.position[0]
            + sample.position[1] * sample.position[1])
            .sqrt();
        let phi = sample.position[2].atan2(radial - scale);
        let ring = (((theta.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU)
            * ring_bins as f32)
            .floor() as usize)
            .min(ring_bins - 1);
        let tube = (((phi.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU)
            * tube_bins as f32)
            .floor() as usize)
            .min(tube_bins - 1);
        covered_rings.insert(ring);
        covered_tubes.insert(tube);
        assert!(
            uv_torus_surface_error(sample.position, scale) <= minor * 0.08,
            "sample {sample_idx} should remain near the torus surface"
        );
    }

    assert!(
        covered_rings.len() >= 20,
        "low sample counts should cover most torus rings, got {}",
        covered_rings.len()
    );
    assert!(
        covered_tubes.len() >= 12,
        "low sample counts should cover most torus tube bins, got {}",
        covered_tubes.len()
    );
}
#[test]
fn mesh_surface_samples_are_area_weighted() {
    let target = TriangleMeshTarget::new(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
        ],
        vec![[0, 1, 2], [0, 3, 4]],
    )
    .unwrap();
    let mut large_triangle_samples = 0usize;
    let samples = 256usize;
    for sample_idx in 0..samples {
        let sample = target.surface_sample(sample_idx);
        if sample.position[0] + sample.position[1] > 1.15 {
            large_triangle_samples += 1;
        }
    }

    assert!(
        large_triangle_samples > samples * 3 / 4,
        "area-weighted sampling should strongly favor the larger triangle, got {large_triangle_samples}/{samples}"
    );
}
#[test]
fn mesh_random_surface_samples_are_area_weighted() {
    let target = TriangleMeshTarget::new(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
        ],
        vec![[0, 1, 2], [0, 3, 4]],
    )
    .unwrap();
    let mut rng = StdRng::seed_from_u64(0x5eed);
    let mut large_triangle_samples = 0usize;
    let samples = 512usize;
    for _ in 0..samples {
        let sample = target.random_surface_sample(&mut rng);
        if sample.position[0] + sample.position[1] > 1.15 {
            large_triangle_samples += 1;
        }
    }

    assert!(
        large_triangle_samples > samples * 3 / 4,
        "random surface sampling should favor area, got {large_triangle_samples}/{samples}"
    );
}
#[test]
fn utah_teapot_mesh_target_exposes_body_spout_handle_and_lid() {
    let scale = 0.72;
    let target = TriangleMeshTarget::utah_teapot(scale).unwrap();

    assert!(target.vertices.len() > 3_000);
    assert!(target.faces.len() > 8_000);
    assert_eq!(target.colors.as_ref().unwrap().len(), target.vertices.len());

    let (bounds_min, bounds_max) = target.bounds();
    assert!(bounds_min[0] < -0.75 * scale);
    assert!(bounds_max[0] > 0.65 * scale);
    assert!(bounds_min[1] < -0.45 * scale);
    assert!(bounds_max[1] > 0.45 * scale);
    assert!(bounds_min[2] < -0.45 * scale);
    assert!(bounds_max[2] > 0.45 * scale);

    let body = target.project([0.0, 0.0, -0.05 * scale]);
    let spout = target.project([0.82 * scale, 0.0, 0.05 * scale]);
    let handle = target.project([-0.92 * scale, 0.0, 0.02 * scale]);
    let lid = target.project([0.0, 0.0, 0.66 * scale]);

    assert!(body.closest[0].abs() < 0.58 * scale);
    assert!(spout.closest[0] > 0.55 * scale);
    assert!(handle.closest[0] < -0.70 * scale);
    assert!(lid.closest[2] > 0.38 * scale);
    for projection in [body, spout, handle, lid] {
        assert!(projection.distance.is_finite());
        assert!(projection.normal.iter().all(|value| value.is_finite()));
        assert!((dot3(projection.normal, projection.normal).sqrt() - 1.0).abs() < 5.0e-3);
        assert!(
            projection
                .color
                .iter()
                .all(|value| (0.0..=1.0).contains(value))
        );
    }
}

#[test]
fn procedural_mesh_targets_are_sampleable_and_colored() {
    for target in [
        TriangleMeshTarget::sphere(0.72, 16).unwrap(),
        TriangleMeshTarget::ellipsoid(0.72, 16).unwrap(),
        TriangleMeshTarget::cube(0.72).unwrap(),
        TriangleMeshTarget::cylinder(0.72, 24).unwrap(),
        TriangleMeshTarget::cone(0.72, 24).unwrap(),
        TriangleMeshTarget::capsule(0.72, 12).unwrap(),
    ] {
        assert!(!target.vertices.is_empty());
        assert!(!target.faces.is_empty());
        assert_eq!(target.colors.as_ref().unwrap().len(), target.vertices.len());
        let (bounds_min, bounds_max) = target.bounds();
        for axis in 0..3 {
            assert!(bounds_min[axis].is_finite());
            assert!(bounds_max[axis].is_finite());
            assert!(bounds_max[axis] > bounds_min[axis]);
        }
        for sample_idx in 0..64 {
            let sample = target.surface_sample(sample_idx);
            let projection = target.project(sample.position);
            assert!(projection.distance.is_finite());
            assert!(
                projection.distance <= 0.12,
                "procedural target sample should project near its source surface"
            );
            assert!((dot3(sample.normal, sample.normal).sqrt() - 1.0).abs() < 5.0e-3);
            assert!(
                sample.color.iter().all(|value| (0.0..=1.0).contains(value)),
                "procedural target color out of range"
            );
        }
    }
}
