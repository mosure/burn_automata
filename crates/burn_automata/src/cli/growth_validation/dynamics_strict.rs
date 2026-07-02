use super::*;

pub(crate) fn apply_material_liveness_strict_check(
    checks: &mut Growth3dStrictChecksReport,
    material_liveness: Growth3dMaterialLivenessReport,
) {
    checks.material_visible_particles_live = material_liveness.passed;
    if !material_liveness.passed
        && !checks
            .failure_reasons
            .contains(&"material_visible_particles_live")
    {
        checks
            .failure_reasons
            .push("material_visible_particles_live");
    }
    checks.passed = checks.failure_reasons.is_empty();
}

pub(crate) fn apply_material_liveness_strict_score(
    score: &mut Growth3dStrictScoreReport,
    material_liveness: Growth3dMaterialLivenessReport,
) {
    let inactive_fraction_penalty = material_liveness.inactive_material_visible_fraction * 10.0;
    let max_inactive_opacity = material_liveness.max_inactive_material_opacity;
    let max_inactive_opacity_penalty = if max_inactive_opacity.is_finite() {
        ((max_inactive_opacity - material_liveness.inactive_material_logit_threshold).max(0.0))
            / 10.0
    } else {
        0.0
    };
    score.material_visible_inactive_fraction = material_liveness.inactive_material_visible_fraction;
    score.material_visible_inactive_fraction_penalty = inactive_fraction_penalty;
    score.material_visible_max_inactive_opacity = max_inactive_opacity;
    score.material_visible_max_inactive_opacity_penalty = max_inactive_opacity_penalty;
    score.score += inactive_fraction_penalty + max_inactive_opacity_penalty;
}

pub(crate) fn apply_temporal_activation_strict_score(
    score: &mut Growth3dStrictScoreReport,
    temporal: &Growth3dTemporalReport,
    rollout_steps: usize,
) {
    let schedule_error = temporal_activation_schedule_error(temporal, rollout_steps);
    let penalty = schedule_error * TEMPORAL_ACTIVATION_SCORE_WEIGHT;
    score.temporal_activation_schedule_error = schedule_error;
    score.temporal_activation_schedule_penalty = penalty;
    score.score += penalty;
}

pub(crate) fn apply_morphogenesis_dynamics_strict_score(
    score: &mut Growth3dStrictScoreReport,
    motion: &Growth3dMotionReport,
    mean_final_displacement: f32,
    seed_scale: f32,
) {
    const MOTION_PEAK_TARGET: f32 = 0.01;
    const ACTIVE_STEP_TARGET: f32 = 0.50;
    const SUSTAINED_STEP_TARGET: f32 = 0.25;

    let displacement_target = growth_3d_seed_radius(seed_scale).max(1.0e-6);
    let peak_penalty = relative_shortfall(MOTION_PEAK_TARGET, motion.peak_mean_dx);
    let active_step_penalty = relative_shortfall(ACTIVE_STEP_TARGET, motion.active_step_fraction);
    let sustained_step_penalty =
        relative_shortfall(SUSTAINED_STEP_TARGET, motion.sustained_step_fraction);
    let displacement_penalty = relative_shortfall(displacement_target, mean_final_displacement);

    score.motion_peak_mean_dx = motion.peak_mean_dx;
    score.motion_peak_penalty = peak_penalty;
    score.motion_active_step_fraction = motion.active_step_fraction;
    score.motion_active_step_penalty = active_step_penalty;
    score.motion_sustained_step_fraction = motion.sustained_step_fraction;
    score.motion_sustained_step_penalty = sustained_step_penalty;
    score.mean_final_displacement = mean_final_displacement;
    score.mean_final_displacement_penalty = displacement_penalty;
    score.score +=
        peak_penalty + active_step_penalty + sustained_step_penalty + displacement_penalty;
}

fn relative_shortfall(target: f32, value: f32) -> f32 {
    if !target.is_finite() || target <= 0.0 {
        return 0.0;
    }
    if !value.is_finite() {
        return 1.0;
    }
    ((target - value) / target).clamp(0.0, 1.0)
}

pub(crate) fn apply_material_visible_surface_tail_strict_check(
    checks: &mut Growth3dStrictChecksReport,
    material_visible_surface_tail: Growth3dSurfaceTailReport,
) {
    let passed = material_visible_surface_tail.p99_distance < GROWTH_3D_SURFACE_MAX_DISTANCE
        && material_visible_surface_tail.over_threshold_fraction <= 0.005
        && material_visible_surface_tail.opacity_weighted_over_threshold_fraction <= 0.005;
    checks.material_visible_surface_tail_bounded = passed;
    if !passed
        && !checks
            .failure_reasons
            .contains(&"material_visible_surface_tail_bounded")
    {
        checks
            .failure_reasons
            .push("material_visible_surface_tail_bounded");
    }
    checks.passed = checks.failure_reasons.is_empty();
}

pub(crate) fn apply_material_visible_surface_tail_strict_score(
    score: &mut Growth3dStrictScoreReport,
    material_visible_surface_tail: Growth3dSurfaceTailReport,
) {
    let p99_penalty =
        (material_visible_surface_tail.p99_distance - GROWTH_3D_SURFACE_MAX_DISTANCE).max(0.0);
    let fraction_penalty = ((material_visible_surface_tail.over_threshold_fraction - 0.005)
        .max(0.0)
        + (material_visible_surface_tail.opacity_weighted_over_threshold_fraction - 0.005)
            .max(0.0))
        * 10.0;
    score.material_visible_surface_tail_p99_distance = material_visible_surface_tail.p99_distance;
    score.material_visible_surface_tail_p99_penalty = p99_penalty;
    score.material_visible_surface_tail_over_threshold_fraction =
        material_visible_surface_tail.over_threshold_fraction;
    score.material_visible_surface_tail_fraction_penalty = fraction_penalty;
    score.score += p99_penalty + fraction_penalty;
}

pub(crate) fn apply_surface_profile_strict_check(
    checks: &mut Growth3dStrictChecksReport,
    active_profile: &SurfaceCoverageProfileReport,
    material_visible_profile: &SurfaceCoverageProfileReport,
) {
    let active_passed = surface_profile_passes_strict_coverage(active_profile);
    let material_visible_passed = surface_profile_passes_strict_coverage(material_visible_profile);
    checks.surface_coverage_profile = active_passed;
    checks.material_visible_surface_coverage_profile = material_visible_passed;
    if !active_passed && !checks.failure_reasons.contains(&"surface_coverage_profile") {
        checks.failure_reasons.push("surface_coverage_profile");
    }
    if !material_visible_passed
        && !checks
            .failure_reasons
            .contains(&"material_visible_surface_coverage_profile")
    {
        checks
            .failure_reasons
            .push("material_visible_surface_coverage_profile");
    }
    checks.passed = checks.failure_reasons.is_empty();
}

pub(crate) fn apply_dynamic_growth_3d_strict_checks(
    checks: &mut Growth3dStrictChecksReport,
    dormant_drift: Growth3dDormantDriftReport,
    material_liveness: Growth3dMaterialLivenessReport,
    material_visible_surface_tail: Growth3dSurfaceTailReport,
    active_profile: &SurfaceCoverageProfileReport,
    material_visible_profile: &SurfaceCoverageProfileReport,
) {
    apply_dormant_drift_strict_check(checks, dormant_drift);
    apply_material_liveness_strict_check(checks, material_liveness);
    apply_material_visible_surface_tail_strict_check(checks, material_visible_surface_tail);
    apply_surface_profile_strict_check(checks, active_profile, material_visible_profile);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_dynamic_growth_3d_strict_score(
    score: &mut Growth3dStrictScoreReport,
    temporal: &Growth3dTemporalReport,
    rollout_steps: usize,
    motion: &Growth3dMotionReport,
    mean_final_displacement: f32,
    seed_scale: f32,
    material_liveness: Growth3dMaterialLivenessReport,
    material_visible_surface_tail: Growth3dSurfaceTailReport,
    active_profile: &SurfaceCoverageProfileReport,
    material_visible_profile: &SurfaceCoverageProfileReport,
) {
    apply_temporal_activation_strict_score(score, temporal, rollout_steps);
    apply_morphogenesis_dynamics_strict_score(score, motion, mean_final_displacement, seed_scale);
    apply_material_liveness_strict_score(score, material_liveness);
    apply_material_visible_surface_tail_strict_score(score, material_visible_surface_tail);
    apply_surface_profile_strict_score(score, active_profile, material_visible_profile);
}

pub(crate) fn surface_profile_passes_strict_coverage(
    profile: &SurfaceCoverageProfileReport,
) -> bool {
    profile.covered_bin_fraction >= GROWTH_3D_MIN_SURFACE_PROFILE_BIN_FRACTION
        && profile.mean_bin_covered_fraction >= GROWTH_3D_MIN_SURFACE_PROFILE_MEAN_BIN_COVERAGE
}

pub(crate) fn apply_surface_profile_strict_score(
    score: &mut Growth3dStrictScoreReport,
    active_profile: &SurfaceCoverageProfileReport,
    material_visible_profile: &SurfaceCoverageProfileReport,
) {
    let active_bin_penalty =
        (GROWTH_3D_MIN_SURFACE_PROFILE_BIN_FRACTION - active_profile.covered_bin_fraction).max(0.0);
    let active_mean_penalty = (GROWTH_3D_MIN_SURFACE_PROFILE_MEAN_BIN_COVERAGE
        - active_profile.mean_bin_covered_fraction)
        .max(0.0);
    let material_bin_penalty = (GROWTH_3D_MIN_SURFACE_PROFILE_BIN_FRACTION
        - material_visible_profile.covered_bin_fraction)
        .max(0.0);
    let material_mean_penalty = (GROWTH_3D_MIN_SURFACE_PROFILE_MEAN_BIN_COVERAGE
        - material_visible_profile.mean_bin_covered_fraction)
        .max(0.0);
    score.surface_covered_bin_fraction = active_profile.covered_bin_fraction;
    score.surface_bin_penalty = active_bin_penalty;
    score.surface_mean_bin_covered_fraction = active_profile.mean_bin_covered_fraction;
    score.surface_coverage_mean_penalty = active_mean_penalty;
    score.material_visible_surface_covered_bin_fraction =
        material_visible_profile.covered_bin_fraction;
    score.material_visible_surface_bin_penalty = material_bin_penalty;
    score.material_visible_surface_mean_bin_covered_fraction =
        material_visible_profile.mean_bin_covered_fraction;
    score.material_visible_surface_mean_penalty = material_mean_penalty;
    score.score +=
        active_bin_penalty + active_mean_penalty + material_bin_penalty + material_mean_penalty;
}
