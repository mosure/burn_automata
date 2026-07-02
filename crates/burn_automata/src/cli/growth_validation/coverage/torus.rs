use super::*;

pub(crate) fn torus_angular_coverage_report(
    positions: &[[f32; 4]],
    states: &[f32],
    state_dims: usize,
    scale: f32,
    threshold: f32,
    ring_bins: usize,
    tube_bins: usize,
) -> TorusAngularCoverageReport {
    let ring_bins = ring_bins.max(1);
    let tube_bins = tube_bins.max(1);
    let major = scale.max(1.0e-4);
    let minor = major * UV_TORUS_MINOR_RATIO;
    let active_positions = positions
        .iter()
        .enumerate()
        .filter_map(|(idx, position)| {
            let opacity = states[idx * state_dims + 3];
            (opacity > -1.0).then_some(*position)
        })
        .collect::<Vec<_>>();
    if active_positions.is_empty() {
        return TorusAngularCoverageReport {
            ring_bins,
            tube_bins,
            threshold,
            covered_joint_bins: 0,
            covered_ring_bins: 0,
            covered_tube_bins: 0,
            joint_coverage_fraction: 0.0,
            ring_coverage_fraction: 0.0,
            tube_coverage_fraction: 0.0,
            max_ring_gap_bins: ring_bins,
            max_tube_gap_bins: tube_bins,
            mean_distance: f32::MAX,
            max_distance: f32::MAX,
        };
    }
    let mut joint_covered = vec![false; ring_bins * tube_bins];
    let mut ring_covered = vec![false; ring_bins];
    let mut tube_covered = vec![false; tube_bins];
    let mut sum_distance = 0.0_f32;
    let mut max_distance = 0.0_f32;

    for ring in 0..ring_bins {
        let theta = std::f32::consts::TAU * (ring as f32 + 0.5) / ring_bins as f32;
        let theta_cos = theta.cos();
        let theta_sin = theta.sin();
        for tube in 0..tube_bins {
            let phi = std::f32::consts::TAU * (tube as f32 + 0.5) / tube_bins as f32;
            let radial = major + minor * phi.cos();
            let sample = [radial * theta_cos, radial * theta_sin, minor * phi.sin()];
            let distance = nearest_position3_distance(sample, &active_positions);
            sum_distance += distance;
            max_distance = max_distance.max(distance);
            if distance <= threshold {
                joint_covered[ring * tube_bins + tube] = true;
                ring_covered[ring] = true;
                tube_covered[tube] = true;
            }
        }
    }

    let total_bins = ring_bins * tube_bins;
    let covered_joint_bins = joint_covered.iter().filter(|covered| **covered).count();
    let covered_ring_bins = ring_covered.iter().filter(|covered| **covered).count();
    let covered_tube_bins = tube_covered.iter().filter(|covered| **covered).count();
    TorusAngularCoverageReport {
        ring_bins,
        tube_bins,
        threshold,
        covered_joint_bins,
        covered_ring_bins,
        covered_tube_bins,
        joint_coverage_fraction: covered_joint_bins as f32 / total_bins.max(1) as f32,
        ring_coverage_fraction: covered_ring_bins as f32 / ring_bins as f32,
        tube_coverage_fraction: covered_tube_bins as f32 / tube_bins as f32,
        max_ring_gap_bins: max_circular_false_run(&ring_covered),
        max_tube_gap_bins: max_circular_false_run(&tube_covered),
        mean_distance: sum_distance / total_bins.max(1) as f32,
        max_distance,
    }
}

fn nearest_position3_distance(sample: [f32; 3], positions: &[[f32; 4]]) -> f32 {
    if positions.is_empty() {
        return f32::MAX;
    }
    positions
        .iter()
        .map(|position| {
            let dx = sample[0] - position[0];
            let dy = sample[1] - position[1];
            let dz = sample[2] - position[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(f32::MAX, f32::min)
}

fn max_circular_false_run(values: &[bool]) -> usize {
    if values.is_empty() || values.iter().all(|value| *value) {
        return 0;
    }
    if values.iter().all(|value| !*value) {
        return values.len();
    }
    let mut max_run = 0usize;
    let mut run = 0usize;
    for idx in 0..values.len() * 2 {
        if values[idx % values.len()] {
            max_run = max_run.max(run);
            run = 0;
        } else {
            run += 1;
            max_run = max_run.max(run.min(values.len()));
        }
    }
    max_run.min(values.len())
}
