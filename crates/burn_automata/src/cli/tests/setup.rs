use super::*;

#[test]
fn train_defaults_to_seeded_growth_target() {
    let seed = default_train_target_seed(AutomataPreset::Growing3dGs, None, false);

    assert_eq!(seed, Some(DEFAULT_GROWTH_TARGET_SEED));
    assert_eq!(
        train_target_source(AutomataPreset::Growing3dGs, seed, false),
        "seeded:Growing3dGs:42"
    );
}

#[test]
fn train_source_defaults_to_rollout_local_metadata() {
    let seed = default_train_target_seed(AutomataPreset::Growing2d, None, false);
    let target_source = train_target_source(AutomataPreset::Growing2d, seed, false);

    assert_eq!(
        training_source_with_batch(TrainingBatchArg::Rollout, &target_source),
        "rollout-local:seeded:Growing2d:42"
    );
    assert_eq!(
        training_source_with_batch(TrainingBatchArg::Features, &target_source),
        "feature-rows:seeded:Growing2d:42"
    );
}

#[test]
fn train_zero_update_requires_explicit_flag() {
    let seed = default_train_target_seed(AutomataPreset::Growing2d, None, true);

    assert_eq!(seed, None);
    assert_eq!(
        train_target_source(AutomataPreset::Growing2d, seed, true),
        "explicit-zero-update"
    );
}

#[test]
fn mesh_training_sources_separate_rollout_local_from_projection_baseline() {
    assert!(UV_TORUS_POSITION_FIELD_TARGET_SOURCE.contains("position-field"));
    assert!(TEAPOT_POSITION_FIELD_TARGET_SOURCE.contains("position-field"));
    assert!(UV_TORUS_ROLLOUT_FIELD_TARGET_SOURCE.contains("rollout-position-field"));
    assert!(TEAPOT_ROLLOUT_FIELD_TARGET_SOURCE.contains("rollout-position-field"));
    assert!(UV_TORUS_MORPHOGEN_ROLLOUT_TARGET_SOURCE.contains("rollout-local"));
    assert!(TEAPOT_MORPHOGEN_ROLLOUT_TARGET_SOURCE.contains("rollout-local"));
    assert!(UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE.contains("conditionless-local"));
    assert!(TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE.contains("conditionless-local"));
    assert!(UV_TORUS_CONDITIONLESS_COMPACT_TARGET_SOURCE.contains("random-ball"));
    assert!(TEAPOT_CONDITIONLESS_COMPACT_TARGET_SOURCE.contains("random-ball"));
    assert!(UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE.contains("substrate"));
    assert!(TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE.contains("substrate"));
    assert!(UV_TORUS_CONDITIONLESS_LOCAL_NOSCAFFOLD_TARGET_SOURCE.contains("no-scaffold"));
    assert!(TEAPOT_CONDITIONLESS_LOCAL_NOSCAFFOLD_TARGET_SOURCE.contains("no-scaffold"));
    assert_eq!(
        mesh_conditionless_local_target_source_for_seed(
            MeshTargetArg::Torus,
            ParticleSeed::Growth3d,
        ),
        UV_TORUS_CONDITIONLESS_COMPACT_TARGET_SOURCE
    );
    assert_eq!(
        mesh_conditionless_local_target_source_for_seed(
            MeshTargetArg::Teapot,
            ParticleSeed::Growth3d,
        ),
        TEAPOT_CONDITIONLESS_COMPACT_TARGET_SOURCE
    );
    assert_eq!(
        mesh_conditionless_local_target_source_for_seed(
            MeshTargetArg::Torus,
            ParticleSeed::LocalSubstrateGrowth3d,
        ),
        UV_TORUS_CONDITIONLESS_LOCAL_NOSCAFFOLD_TARGET_SOURCE
    );
    assert_eq!(
        mesh_conditionless_local_target_source_for_seed(
            MeshTargetArg::Teapot,
            ParticleSeed::LocalSubstrateGrowth3d,
        ),
        TEAPOT_CONDITIONLESS_LOCAL_NOSCAFFOLD_TARGET_SOURCE
    );
    assert_eq!(
        mesh_conditionless_local_target_source_for_seed(
            MeshTargetArg::Torus,
            ParticleSeed::TorusGrowth3d,
        ),
        UV_TORUS_CONDITIONLESS_COMPACT_TARGET_SOURCE
    );
    assert_eq!(
        mesh_conditionless_local_target_source_for_seed(
            MeshTargetArg::Torus,
            ParticleSeed::TorusSubstrateGrowth3d,
        ),
        UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE
    );
    assert_eq!(
        mesh_conditionless_local_target_source_for_seed(
            MeshTargetArg::Torus,
            ParticleSeed::TorusLocalSubstrateGrowth3d,
        ),
        UV_TORUS_CONDITIONLESS_LOCAL_NOSCAFFOLD_TARGET_SOURCE
    );
    assert!(UV_TORUS_MORPHOGEN_BASELINE_TARGET_SOURCE.contains("seed-frame"));
    assert!(TEAPOT_MORPHOGEN_BASELINE_TARGET_SOURCE.contains("seed-frame"));
}

#[test]
fn train_render3d_defaults_to_full_coverage_adjoint_with_opt_out() {
    assert!(app::resolve_full_coverage_adjoint(false, false).unwrap());
    assert!(app::resolve_full_coverage_adjoint(true, false).unwrap());
    assert!(!app::resolve_full_coverage_adjoint(false, true).unwrap());
    assert!(app::resolve_full_coverage_adjoint(true, true).is_err());
}

#[test]
fn train_render3d_defaults_to_selection_seed_direct_training_with_opt_out() {
    assert!(app::resolve_direct_selection_seed_training(false, false).unwrap());
    assert!(app::resolve_direct_selection_seed_training(true, false).unwrap());
    assert!(!app::resolve_direct_selection_seed_training(false, true).unwrap());
    assert!(app::resolve_direct_selection_seed_training(true, true).is_err());
}

#[test]
fn train_render3d_defaults_to_strict_line_search_with_value_opt_out() {
    let args = CliArgs::try_parse_from(["burn_automata", "train-render3d"]).unwrap();
    let Command::TrainRender3d {
        direct_line_search, ..
    } = args.command
    else {
        panic!("expected train-render3d command");
    };
    assert!(
        direct_line_search,
        "direct rollout training should default to strict-score line search"
    );

    let args = CliArgs::try_parse_from(["burn_automata", "train-render3d"]).unwrap();
    let Command::TrainRender3d {
        direct_line_search_scales,
        ..
    } = args.command
    else {
        panic!("expected train-render3d command");
    };
    assert_eq!(
        &direct_line_search_scales[..4],
        &[0.0625, 0.125, 0.25, 0.5],
        "direct line search should include fine continuation scales after an activation breakthrough"
    );
    assert!(
        direct_line_search_scales.contains(&1.0),
        "default line search should still include the nominal step"
    );

    let args = CliArgs::try_parse_from(["burn_automata", "train-render3d"]).unwrap();
    let Command::TrainRender3d { grad_clip_norm, .. } = args.command else {
        panic!("expected train-render3d command");
    };
    assert_eq!(
        grad_clip_norm, 1.0,
        "row-normalized direct rollout gradients should use the generic SGD clip default"
    );

    let args = CliArgs::try_parse_from([
        "burn_automata",
        "train-render3d",
        "--direct-line-search=false",
    ])
    .unwrap();
    let Command::TrainRender3d {
        direct_line_search, ..
    } = args.command
    else {
        panic!("expected train-render3d command");
    };
    assert!(
        !direct_line_search,
        "explicit false should keep single-step direct training available as an ablation"
    );
}

#[test]
fn train_render3d_defaults_to_shared_base_adapter_updates() {
    let args = CliArgs::try_parse_from(["burn_automata", "train-render3d"]).unwrap();
    let Command::TrainRender3d {
        weight_update_mode,
        adapter_rank,
        adapter_alpha,
        ..
    } = args.command
    else {
        panic!("expected train-render3d command");
    };
    assert_eq!(
        weight_update_mode,
        RenderWeightUpdateModeArg::Adapter,
        "3D render training should prefer shared-base low-rank object adapters"
    );
    assert_eq!(adapter_rank, 8);
    assert_eq!(adapter_alpha, 8.0);

    let args = CliArgs::try_parse_from([
        "burn_automata",
        "train-render3d",
        "--weight-update-mode",
        "full",
        "--adapter-rank",
        "4",
        "--adapter-alpha",
        "2.5",
    ])
    .unwrap();
    let Command::TrainRender3d {
        weight_update_mode,
        adapter_rank,
        adapter_alpha,
        ..
    } = args.command
    else {
        panic!("expected train-render3d command");
    };
    assert_eq!(weight_update_mode, RenderWeightUpdateModeArg::Full);
    assert_eq!(adapter_rank, 4);
    assert_eq!(adapter_alpha, 2.5);
}

#[test]
fn train_render3d_adapter_suite_defaults_to_shared_base_sweep() {
    let args = CliArgs::try_parse_from([
        "burn_automata",
        "train-render3d-adapters",
        "--base-model",
        "artifacts/shared_base.bpk",
    ])
    .unwrap();
    let Command::TrainRender3dAdapters {
        base_model,
        targets,
        output_dir,
        training_backend,
        adapter_rank,
        adapter_alpha,
        particles,
        ..
    } = args.command
    else {
        panic!("expected train-render3d-adapters command");
    };
    assert_eq!(base_model, PathBuf::from("artifacts/shared_base.bpk"));
    assert_eq!(targets, vec![MeshTargetArg::Torus, MeshTargetArg::Teapot]);
    assert_eq!(
        output_dir,
        PathBuf::from("artifacts/render_3d_adapter_suite")
    );
    assert_eq!(training_backend, RenderTrainingBackendArg::DirectRollout);
    assert_eq!(adapter_rank, 8);
    assert_eq!(adapter_alpha, 8.0);
    assert_eq!(particles, 512);
}

#[test]
fn train_render3d_exposes_robust_geometry_objective_gains() {
    let args = CliArgs::try_parse_from(["burn_automata", "train-render3d"]).unwrap();
    let Command::TrainRender3d {
        coverage_gain,
        coverage_samples,
        coverage_repulsion_gain,
        coverage_normal_weight,
        extent_gain,
        surface_gain,
        surface_escape_gain,
        ..
    } = args.command
    else {
        panic!("expected train-render3d command");
    };
    assert_eq!(coverage_gain, ROBUST_3D_COVERAGE_GAIN);
    assert_eq!(coverage_samples, ROBUST_3D_COVERAGE_SAMPLES);
    assert_eq!(coverage_repulsion_gain, ROBUST_3D_COVERAGE_REPULSION_GAIN);
    assert_eq!(coverage_normal_weight, ROBUST_3D_COVERAGE_NORMAL_WEIGHT);
    assert_eq!(extent_gain, ROBUST_3D_EXTENT_GAIN);
    assert_eq!(surface_gain, ROBUST_3D_SURFACE_GAIN);
    assert_eq!(surface_escape_gain, ROBUST_3D_SURFACE_ESCAPE_GAIN);

    let args = CliArgs::try_parse_from([
        "burn_automata",
        "train-render3d",
        "--coverage-gain",
        "0.12",
        "--extent-gain",
        "0.34",
        "--surface-gain",
        "0.056",
    ])
    .unwrap();
    let Command::TrainRender3d {
        coverage_gain,
        extent_gain,
        surface_gain,
        ..
    } = args.command
    else {
        panic!("expected train-render3d command");
    };
    assert_eq!(coverage_gain, 0.12);
    assert_eq!(extent_gain, 0.34);
    assert_eq!(surface_gain, 0.056);
}

#[test]
fn train_render3d_exposes_material_gains_separately_from_opacity_gain() {
    let args = CliArgs::try_parse_from(["burn_automata", "train-render3d"]).unwrap();
    let Command::TrainRender3d {
        opacity_gain,
        liveness_update_multiplier,
        material_liveness_gain,
        material_tail_gain,
        material_suppression_update_multiplier,
        direct_output_gradient_rms_cap,
        ..
    } = args.command
    else {
        panic!("expected train-render3d command");
    };
    assert_eq!(opacity_gain, ROBUST_3D_OPACITY_GAIN);
    assert_eq!(
        liveness_update_multiplier,
        ROBUST_3D_LIVENESS_UPDATE_MULTIPLIER
    );
    assert_eq!(material_liveness_gain, ROBUST_3D_MATERIAL_LIVENESS_GAIN);
    assert_eq!(material_tail_gain, ROBUST_3D_MATERIAL_TAIL_GAIN);
    assert_eq!(
        material_suppression_update_multiplier,
        ROBUST_3D_MATERIAL_SUPPRESSION_UPDATE_MULTIPLIER
    );
    assert_eq!(
        direct_output_gradient_rms_cap,
        ROBUST_3D_DIRECT_OUTPUT_GRADIENT_RMS_CAP
    );

    let args = CliArgs::try_parse_from([
        "burn_automata",
        "train-render3d",
        "--opacity-gain",
        "0",
        "--material-liveness-gain",
        "0.35",
        "--material-tail-gain",
        "0.45",
        "--liveness-update-multiplier",
        "7.5",
        "--material-suppression-update-multiplier",
        "3.5",
        "--direct-output-gradient-rms-cap",
        "0.125",
    ])
    .unwrap();
    let Command::TrainRender3d {
        opacity_gain,
        liveness_update_multiplier,
        material_liveness_gain,
        material_tail_gain,
        material_suppression_update_multiplier,
        direct_output_gradient_rms_cap,
        ..
    } = args.command
    else {
        panic!("expected train-render3d command");
    };
    assert_eq!(opacity_gain, 0.0);
    assert_eq!(liveness_update_multiplier, 7.5);
    assert_eq!(material_liveness_gain, 0.35);
    assert_eq!(material_tail_gain, 0.45);
    assert_eq!(material_suppression_update_multiplier, 3.5);
    assert_eq!(direct_output_gradient_rms_cap, 0.125);
}

#[test]
fn render_training_source_preserves_local_refinement_lineage() {
    let local_source = render_training_source(
        MeshTargetArg::Torus,
        Some(UV_TORUS_CONDITIONLESS_COMPACT_TARGET_SOURCE),
        ParticleSeed::TorusGrowth3d,
    );
    assert!(local_source.starts_with("render-refined-rust:"));
    assert!(local_source.contains("conditionless-local"));
    assert!(!local_source.contains("position-field"));
    assert!(!local_source.contains("render-proxy-rust"));

    let already_refined_source = render_training_source(
        MeshTargetArg::Torus,
        Some(&local_source),
        ParticleSeed::TorusGrowth3d,
    );
    assert_eq!(already_refined_source, local_source);

    let field_source = render_training_source(
        MeshTargetArg::Torus,
        Some(UV_TORUS_POSITION_FIELD_TARGET_SOURCE),
        ParticleSeed::TorusFieldDense3d,
    );
    assert!(field_source.starts_with("render-proxy-rust:"));
    assert!(field_source.contains("position-field"));

    let default_source = render_training_source(
        MeshTargetArg::Teapot,
        None,
        ParticleSeed::TeapotFieldDense3d,
    );
    assert!(default_source.contains("field-baseline"));
}

#[test]
fn render_adapter_training_source_marks_target_adapter_without_proxy_lineage() {
    let source = render_adapter_training_source(
        MeshTargetArg::Torus,
        Some("shared-3d-base:conditionless-local"),
        ParticleSeed::LocalSubstrateGrowth3d,
    );

    assert!(source.starts_with("render-adapter-rust:"));
    assert!(source.contains("shared-base=shared-3d-base:conditionless-local"));
    assert!(!source.contains("render-proxy-rust"));
    assert!(target_conditionless_lineage(MeshTargetArg::Torus, &source));
    assert!(target_seed_conditionless_lineage(
        MeshTargetArg::Torus,
        ParticleSeed::LocalSubstrateGrowth3d,
        &source
    ));
    assert!(!target_conditionless_lineage(
        MeshTargetArg::Teapot,
        &source
    ));

    let teapot_source = render_adapter_training_source(
        MeshTargetArg::Teapot,
        Some("shared-3d-base:conditionless-local"),
        ParticleSeed::LocalSubstrateGrowth3d,
    );
    assert!(target_seed_conditionless_lineage(
        MeshTargetArg::Teapot,
        ParticleSeed::LocalSubstrateGrowth3d,
        &teapot_source
    ));
}

#[test]
fn render_training_source_does_not_preserve_mismatched_target_lineage() {
    assert!(target_conditionless_lineage(
        MeshTargetArg::Torus,
        UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE
    ));
    assert!(target_conditionless_lineage(
        MeshTargetArg::Teapot,
        TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE
    ));
    assert!(!target_conditionless_lineage(
        MeshTargetArg::Teapot,
        UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE
    ));
    assert!(!target_conditionless_lineage(
        MeshTargetArg::Torus,
        TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE
    ));
    assert!(target_seed_conditionless_lineage(
        MeshTargetArg::Teapot,
        ParticleSeed::LocalSubstrateGrowth3d,
        TEAPOT_CONDITIONLESS_LOCAL_NOSCAFFOLD_TARGET_SOURCE
    ));
    assert!(target_seed_conditionless_lineage(
        MeshTargetArg::Torus,
        ParticleSeed::LocalSubstrateGrowth3d,
        UV_TORUS_CONDITIONLESS_LOCAL_NOSCAFFOLD_TARGET_SOURCE
    ));
    assert!(target_seed_conditionless_lineage(
        MeshTargetArg::Teapot,
        ParticleSeed::TeapotLocalSubstrateGrowth3d,
        TEAPOT_CONDITIONLESS_LOCAL_NOSCAFFOLD_TARGET_SOURCE
    ));
    assert!(!target_seed_conditionless_lineage(
        MeshTargetArg::Teapot,
        ParticleSeed::TeapotLocalSubstrateGrowth3d,
        TEAPOT_CONDITIONLESS_COMPACT_TARGET_SOURCE
    ));

    let mismatched_source = render_training_source(
        MeshTargetArg::Teapot,
        Some(UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE),
        ParticleSeed::TeapotLocalSubstrateGrowth3d,
    );
    assert!(
        mismatched_source.starts_with("render-proxy-rust:"),
        "target-mismatched local lineage must not be preserved as a render refinement"
    );
    assert!(mismatched_source.contains("Teapot"));
    assert!(mismatched_source.contains(UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE));

    let mismatched_seed_source = render_training_source(
        MeshTargetArg::Teapot,
        Some(TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE),
        ParticleSeed::TorusLocalSubstrateGrowth3d,
    );
    assert!(
        mismatched_seed_source.starts_with("render-proxy-rust:"),
        "target-local lineage must not be preserved when the seed family belongs to another target"
    );
    assert!(mismatched_seed_source.contains("Teapot"));
    assert!(mismatched_seed_source.contains("TorusLocalSubstrateGrowth3d"));

    let mismatched_seed_topology_source = render_training_source(
        MeshTargetArg::Teapot,
        Some(TEAPOT_CONDITIONLESS_COMPACT_TARGET_SOURCE),
        ParticleSeed::TeapotLocalSubstrateGrowth3d,
    );
    assert!(
        mismatched_seed_topology_source.starts_with("render-proxy-rust:"),
        "target-local lineage must not be preserved when source topology is random-ball but seed mode is no-scaffold substrate"
    );
    assert!(mismatched_seed_topology_source.contains("Teapot"));
    assert!(mismatched_seed_topology_source.contains(TEAPOT_CONDITIONLESS_COMPACT_TARGET_SOURCE));
}

#[test]
fn render_training_defaults_match_model_family() {
    assert_eq!(
        render_training_default_seed_mode(MeshTargetArg::Torus),
        ParticleSeed::LocalSubstrateGrowth3d
    );
    assert_eq!(
        render_training_default_seed_mode(MeshTargetArg::Teapot),
        ParticleSeed::LocalSubstrateGrowth3d
    );

    let local_model = NpaModel::seeded(NpaConfig::growing_3dgs(), 7);
    assert_eq!(
        default_render_training_seed_mode(MeshTargetArg::Torus, &local_model),
        ParticleSeed::LocalSubstrateGrowth3d
    );
    assert_eq!(
        default_render_training_seed_mode(MeshTargetArg::Teapot, &local_model),
        ParticleSeed::LocalSubstrateGrowth3d
    );

    let field_model = NpaModel::seeded(NpaConfig::torus_field_3dgs(), 7);
    assert_eq!(
        default_render_training_seed_mode(MeshTargetArg::Torus, &field_model),
        ParticleSeed::TorusFieldDense3d
    );
    assert_eq!(
        default_render_training_seed_mode(MeshTargetArg::Teapot, &field_model),
        ParticleSeed::TeapotFieldDense3d
    );
}

#[test]
fn mesh_target_training_profiles_are_explicit_per_target() {
    let torus = mesh_target_training_profile(MeshTargetArg::Torus);
    assert_eq!(torus.target, MeshTargetArg::Torus);
    assert_eq!(torus.field_scale, UV_TORUS_FIELD_SCALE);
    assert_eq!(torus.render_training_scale, UV_TORUS_RENDER_TRAINING_SCALE);
    assert_eq!(torus.field_seed_mode, ParticleSeed::TorusFieldDense3d);
    assert_eq!(
        torus.conditionless_local_seed_mode,
        ParticleSeed::LocalSubstrateGrowth3d
    );
    assert_eq!(torus.field_motion_gain, UV_TORUS_FIELD_MOTION_GAIN);
    assert_eq!(torus.field_color_gain, UV_TORUS_FIELD_COLOR_GAIN);
    assert_eq!(torus.local_motion_gain, LOCAL_TORUS_MOTION_GAIN);
    assert_eq!(torus.local_color_gain, LOCAL_TORUS_COLOR_GAIN);
    assert_eq!(
        torus.position_field_target_source,
        UV_TORUS_POSITION_FIELD_TARGET_SOURCE
    );
    assert_eq!(
        torus.rollout_field_target_source,
        UV_TORUS_ROLLOUT_FIELD_TARGET_SOURCE
    );
    assert_eq!(
        torus.morphogen_baseline_target_source,
        UV_TORUS_MORPHOGEN_BASELINE_TARGET_SOURCE
    );
    assert_eq!(
        torus.morphogen_rollout_target_source,
        UV_TORUS_MORPHOGEN_ROLLOUT_TARGET_SOURCE
    );
    assert_eq!(
        torus.conditionless_local_target_source,
        UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE
    );
    assert_eq!(torus.lineage_marker, "uv-torus-3d");

    let teapot = mesh_target_training_profile(MeshTargetArg::Teapot);
    assert_eq!(teapot.target, MeshTargetArg::Teapot);
    assert_eq!(teapot.field_scale, DEFAULT_3D_MESH_FIELD_SCALE);
    assert_eq!(teapot.render_training_scale, TEAPOT_RENDER_TRAINING_SCALE);
    assert_eq!(teapot.field_seed_mode, ParticleSeed::TeapotFieldDense3d);
    assert_eq!(
        teapot.conditionless_local_seed_mode,
        ParticleSeed::LocalSubstrateGrowth3d
    );
    assert_eq!(teapot.field_motion_gain, TEAPOT_FIELD_MOTION_GAIN);
    assert_eq!(teapot.field_color_gain, TEAPOT_FIELD_COLOR_GAIN);
    assert_eq!(teapot.local_motion_gain, LOCAL_TEAPOT_MOTION_GAIN);
    assert_eq!(teapot.local_color_gain, LOCAL_TEAPOT_COLOR_GAIN);
    assert_eq!(
        teapot.position_field_target_source,
        TEAPOT_POSITION_FIELD_TARGET_SOURCE
    );
    assert_eq!(
        teapot.rollout_field_target_source,
        TEAPOT_ROLLOUT_FIELD_TARGET_SOURCE
    );
    assert_eq!(
        teapot.morphogen_baseline_target_source,
        TEAPOT_MORPHOGEN_BASELINE_TARGET_SOURCE
    );
    assert_eq!(
        teapot.morphogen_rollout_target_source,
        TEAPOT_MORPHOGEN_ROLLOUT_TARGET_SOURCE
    );
    assert_eq!(
        teapot.conditionless_local_target_source,
        TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE
    );
    assert_eq!(teapot.lineage_marker, "utah-teapot-2026");
    assert_ne!(
        teapot.field_seed_mode, torus.field_seed_mode,
        "generic profile lookup must not reuse torus seed modes for teapot"
    );
    assert_eq!(
        teapot.conditionless_local_seed_mode, torus.conditionless_local_seed_mode,
        "strict local training should share object-agnostic seed modes across mesh targets"
    );
    assert_ne!(
        teapot.conditionless_local_target_source, torus.conditionless_local_target_source,
        "target profile sources must identify the actual mesh target"
    );
    assert_ne!(
        teapot.morphogen_rollout_target_source, torus.morphogen_rollout_target_source,
        "target profile rollout sources must identify the actual mesh target"
    );
}

#[test]
fn mesh_seed_modes_use_neutral_3d_reference_scale() {
    for seed_mode in [
        ParticleSeed::Growth3d,
        ParticleSeed::SubstrateGrowth3d,
        ParticleSeed::LocalGrowth3d,
        ParticleSeed::LocalSubstrateGrowth3d,
        ParticleSeed::TorusFieldDense3d,
        ParticleSeed::TeapotFieldDense3d,
        ParticleSeed::TorusGrowth3d,
        ParticleSeed::TeapotGrowth3d,
        ParticleSeed::TorusLocalSubstrateGrowth3d,
        ParticleSeed::TeapotLocalSubstrateGrowth3d,
    ] {
        assert_eq!(
            reference_seed_scale_for_seed_mode(AutomataPreset::Growing3dGs, seed_mode),
            DEFAULT_3D_MESH_FIELD_SCALE
        );
    }
}

#[test]
fn train_render3d_uses_target_specific_seed_scale_defaults() {
    assert_eq!(
        mesh_target_render_training_seed_scale(MeshTargetArg::Torus),
        UV_TORUS_RENDER_TRAINING_SCALE
    );
    assert_eq!(
        mesh_target_render_training_seed_scale(MeshTargetArg::Teapot),
        TEAPOT_RENDER_TRAINING_SCALE
    );

    let args = CliArgs::try_parse_from(["burn_automata", "train-render3d"]).unwrap();
    let Command::TrainRender3d { seed_scale, .. } = args.command else {
        panic!("expected train-render3d command");
    };
    assert_eq!(
        seed_scale, None,
        "omitted seed scale should be resolved from the target at execution time"
    );

    let args = CliArgs::try_parse_from([
        "burn_automata",
        "train-render3d",
        "--target",
        "torus",
        "--seed-scale",
        "0.9",
    ])
    .unwrap();
    let Command::TrainRender3d { seed_scale, .. } = args.command else {
        panic!("expected train-render3d command");
    };
    assert_eq!(
        seed_scale,
        Some(0.9),
        "explicit seed scale should remain an ablation override"
    );
}

#[test]
fn render_training_validation_extra_seeds_dedupe_selection_set() {
    assert_eq!(
        render_training_validation_extra_seeds(42, &[99, 42, 7, 99]),
        vec![42, 99, 7]
    );
}

#[test]
fn render_training_default_extra_selection_seeds_include_catalog_heldouts() {
    assert_eq!(
        render_training_default_extra_selection_seeds(CATALOG_3D_APP_EVAL_SEED, &[]),
        vec![42, 99],
        "default train-render3d should optimize the app held-out seeds that catalog promotion validates"
    );
    assert_eq!(
        render_training_default_extra_selection_seeds(42, &[99, 7, 42, CATALOG_3D_APP_EVAL_SEED]),
        vec![99, 7, CATALOG_3D_APP_EVAL_SEED],
        "selection seed should stay singular while user extras are deduped after held-outs"
    );
}

#[test]
fn catalog_promotion_validation_extra_seeds_include_app_heldouts() {
    assert_eq!(
        catalog_promotion_validation_extra_seeds(CATALOG_3D_APP_EVAL_SEED, &[]),
        vec![42, 99]
    );
    assert_eq!(
        catalog_promotion_validation_extra_seeds(42, &[99, 7, CATALOG_3D_APP_EVAL_SEED]),
        vec![42, 99, 7]
    );
}

#[test]
fn catalog_promotion_validation_configs_match_app_scale() {
    let mut render = RenderLossConfig {
        image_size: 8,
        target_samples: 0,
        ..RenderLossConfig::default()
    };
    render.world_scale = 0.5;

    let configs = catalog_promotion_validation_configs(
        7,
        &[99],
        0.54,
        ParticleSeed::LocalSubstrateGrowth3d,
        render,
    );

    assert_eq!(configs.len(), 2);
    assert_eq!(
        configs.iter().map(|cfg| cfg.steps).collect::<Vec<_>>(),
        CATALOG_3D_PROMOTION_STEPS.to_vec()
    );
    for cfg in configs {
        assert_eq!(cfg.particle_count, CATALOG_3D_VALIDATION_PARTICLES);
        assert_eq!(cfg.seed, CATALOG_3D_APP_EVAL_SEED);
        assert_eq!(cfg.extra_seeds, vec![42, 99, 7]);
        assert_eq!(cfg.seed_mode, ParticleSeed::LocalSubstrateGrowth3d);
        assert!(
            !growth_3d_seed_has_coordinate_scaffold(cfg.seed_mode),
            "catalog promotion validation must run the same no-scaffold seed family required by the strict gate"
        );
        assert!(matches!(cfg.gate, Growth3dValidationGateArg::Strict));
        assert_eq!(cfg.render.image_size, CATALOG_3D_VALIDATION_IMAGE_SIZE);
        assert_eq!(
            cfg.render.target_samples,
            CATALOG_3D_VALIDATION_TARGET_SAMPLES
        );
        assert_eq!(cfg.render.world_scale, 0.5);
    }
}

#[test]
fn catalog_bound_render_training_uses_app_scale_particle_and_step_floor() {
    let catalog_path = Path::new("assets/models/uv_torus_growth_3d.bpk");
    let diagnostic_path = Path::new("artifacts/uv_torus_growth_3d.bpk");

    assert_eq!(
        render_training_particle_count_for_output(catalog_path, 512),
        CATALOG_3D_VALIDATION_PARTICLES
    );
    assert_eq!(
        render_training_rollout_steps_for_output(catalog_path, 32),
        CATALOG_3D_PROMOTION_STEPS.iter().copied().max().unwrap()
    );
    assert_eq!(
        render_training_particle_count_for_output(diagnostic_path, 512),
        512
    );
    assert_eq!(
        render_training_rollout_steps_for_output(diagnostic_path, 32),
        32
    );
    assert_eq!(
        render_training_particle_count_for_output(catalog_path, 4096),
        4096
    );
    assert_eq!(
        render_training_rollout_steps_for_output(catalog_path, 128),
        128
    );
}

#[test]
fn catalog_bound_render_training_rejects_mismatched_target_lineage() {
    let catalog_path = Path::new("assets/models/teapot_growth_3d.bpk");
    let mismatch = validate_catalog_bound_render_training_output(
        catalog_path,
        MeshTargetArg::Teapot,
        ParticleSeed::LocalSubstrateGrowth3d,
        Some(UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE),
    )
    .unwrap_err()
    .to_string();
    assert!(mismatch.contains("requires a conditionless-local base model for target Teapot"));

    validate_catalog_bound_render_training_output(
        catalog_path,
        MeshTargetArg::Teapot,
        ParticleSeed::LocalSubstrateGrowth3d,
        Some(TEAPOT_CONDITIONLESS_LOCAL_NOSCAFFOLD_TARGET_SOURCE),
    )
    .unwrap();
}

#[test]
fn render_training_base_defaults_to_conditionless_local_growth() {
    let target = uv_torus_mesh_target(UV_TORUS_FIELD_SCALE);
    let (model, source) = render_training_base_model(
        MeshTargetArg::Torus,
        &target,
        render_training_default_seed_mode(MeshTargetArg::Torus),
    )
    .unwrap();

    assert!(!model.config.position_features);
    assert!(local_conditionless_lineage(&source));
    assert!(source.starts_with("ablation-rust:"));
    assert!(source.contains("conditionless-local"));
    assert!(source.contains("substrate"));
    assert!(source.contains("no-scaffold"));
    assert!(!source.contains("position-field"));
    assert!(!source.contains("render-proxy-rust"));

    let err = render_training_base_model(
        MeshTargetArg::Torus,
        &target,
        ParticleSeed::TorusFieldDense3d,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("strict conditionless-local growth seed"));

    let scaffold_err =
        render_training_base_model(MeshTargetArg::Torus, &target, ParticleSeed::TorusGrowth3d)
            .unwrap_err()
            .to_string();
    assert!(
        scaffold_err.contains("strict conditionless-local growth seed"),
        "default render training must not silently fall back to scaffolded compact seeds"
    );
}

#[test]
fn sparse_growth_seed_modes_do_not_preload_target_state() {
    let config = NpaConfig::growing_3dgs();
    for seed_mode in [
        ParticleSeed::Growth3d,
        ParticleSeed::TorusGrowth3d,
        ParticleSeed::TeapotGrowth3d,
    ] {
        let (_positions, states) = seed_particles_scaled(
            1,
            512,
            config.state_dims,
            config.spatial_dims,
            0x5eed,
            seed_mode,
            UV_TORUS_FIELD_SCALE,
        );
        let mut active = 0usize;
        let mut inactive = 0usize;
        for state in states.chunks_exact(config.state_dims) {
            if state[3] > -1.0 {
                active += 1;
            } else {
                inactive += 1;
            }
        }
        let non_opacity_seed_abs_max =
            growth_3d_non_scaffold_seed_abs_max(config.state_dims, seed_mode, &states);

        assert!(active > 0, "{seed_mode:?} should seed a sparse active core");
        assert!(
            inactive > active,
            "{seed_mode:?} should leave most particles dormant"
        );
        assert_eq!(
            non_opacity_seed_abs_max, 0.0,
            "{seed_mode:?} must not preload residual, normal, color, or other target state outside the coordinate scaffold"
        );
    }
}

#[test]
fn local_growth_seed_modes_do_not_write_coordinate_scaffold() {
    let config = NpaConfig::growing_3dgs();
    let seed_modes = [
        ParticleSeed::LocalGrowth3d,
        ParticleSeed::LocalSubstrateGrowth3d,
        ParticleSeed::TorusLocalGrowth3d,
        ParticleSeed::TeapotLocalGrowth3d,
        ParticleSeed::TorusLocalSubstrateGrowth3d,
        ParticleSeed::TeapotLocalSubstrateGrowth3d,
    ];
    for seed_mode in seed_modes {
        let (_positions, states) = seed_particles_scaled(
            1,
            512,
            config.state_dims,
            config.spatial_dims,
            0x5eed,
            seed_mode,
            UV_TORUS_FIELD_SCALE,
        );
        let non_opacity_seed_abs_max =
            growth_3d_non_scaffold_seed_abs_max(config.state_dims, seed_mode, &states);

        assert!(
            !growth_3d_seed_has_coordinate_scaffold(seed_mode),
            "{seed_mode:?} should be eligible for the strict no-scaffold gate"
        );
        assert_eq!(
            non_opacity_seed_abs_max, 0.0,
            "{seed_mode:?} must leave non-opacity state neutral at initialization"
        );
    }
}

#[test]
fn catalog_bound_render_training_requires_local_growth_lineage() {
    assert!(is_catalog_model_output_path(Path::new(
        "assets/models/teapot_growth_3d.bpk"
    )));
    assert!(!is_catalog_model_output_path(Path::new(
        "artifacts/render_trained_3d.bpk"
    )));

    validate_catalog_bound_render_training_output(
        Path::new("artifacts/render_trained_3d.bpk"),
        MeshTargetArg::Teapot,
        ParticleSeed::TeapotFieldDense3d,
        None,
    )
    .unwrap();

    let local_source =
        format!("render-refined-rust:{TEAPOT_CONDITIONLESS_LOCAL_NOSCAFFOLD_TARGET_SOURCE}");
    validate_catalog_bound_render_training_output(
        Path::new("assets/models/teapot_growth_3d.bpk"),
        MeshTargetArg::Teapot,
        ParticleSeed::LocalSubstrateGrowth3d,
        Some(&local_source),
    )
    .unwrap();

    let field_seed_error = validate_catalog_bound_render_training_output(
        Path::new("assets/models/render_trained_3d.bpk"),
        MeshTargetArg::Teapot,
        ParticleSeed::TeapotFieldDense3d,
        Some(&local_source),
    )
    .unwrap_err();
    assert!(field_seed_error.to_string().contains("local growth seed"));

    let source_seed_mismatch_error = validate_catalog_bound_render_training_output(
        Path::new("assets/models/render_trained_3d.bpk"),
        MeshTargetArg::Teapot,
        ParticleSeed::LocalSubstrateGrowth3d,
        Some(TEAPOT_CONDITIONLESS_COMPACT_TARGET_SOURCE),
    )
    .unwrap_err();
    assert!(
        source_seed_mismatch_error
            .to_string()
            .contains("matching seed_mode"),
        "catalog promotion should reject a random-ball source when evaluating a no-scaffold substrate seed"
    );

    let scaffold_seed_error = validate_catalog_bound_render_training_output(
        Path::new("assets/models/render_trained_3d.bpk"),
        MeshTargetArg::Teapot,
        ParticleSeed::TeapotGrowth3d,
        Some(&local_source),
    )
    .unwrap_err();
    assert!(
        scaffold_seed_error
            .to_string()
            .contains("no-scaffold local growth seed"),
        "catalog promotion should fail before training when the seed mode cannot satisfy the strict no-scaffold gate"
    );

    let shortcut_lineage_error = validate_catalog_bound_render_training_output(
        Path::new("assets/models/render_trained_3d.bpk"),
        MeshTargetArg::Teapot,
        ParticleSeed::LocalSubstrateGrowth3d,
        Some(TEAPOT_POSITION_FIELD_TARGET_SOURCE),
    )
    .unwrap_err();
    assert!(
        shortcut_lineage_error
            .to_string()
            .contains("conditionless-local")
    );
}

#[test]
fn catalog_bound_render_training_uses_target_temp_candidate_path() {
    let torus = catalog_bound_candidate_path(MeshTargetArg::Torus, 1234);
    let teapot = catalog_bound_candidate_path(MeshTargetArg::Teapot, 1234);

    assert!(torus.starts_with("target"));
    assert!(teapot.starts_with("target"));
    assert!(!is_catalog_model_output_path(&torus));
    assert!(!is_catalog_model_output_path(&teapot));
    assert!(torus.to_string_lossy().contains("torus"));
    assert!(teapot.to_string_lossy().contains("teapot"));
    assert_ne!(torus, teapot);
}

#[test]
fn diagnostic_3d_outputs_refuse_catalog_paths() {
    validate_diagnostic_3d_output_not_catalog(
        Path::new("target/teapot_probe.bpk"),
        "ablate-local-3d",
    )
    .unwrap();
    validate_diagnostic_3d_output_not_catalog(
        Path::new("artifacts/teapot_probe.bpk"),
        "ablate-local-3d",
    )
    .unwrap();

    let err = validate_diagnostic_3d_output_not_catalog(
        Path::new("assets/models/teapot_probe.bpk"),
        "ablate-local-3d",
    )
    .unwrap_err();
    let message = err.to_string();
    assert!(message.contains("diagnostic 3D artifacts"));
    assert!(message.contains("validate_3d_catalog.py"));
}

#[test]
fn local_3d_continuation_accepts_only_conditionless_local_lineage() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model = NpaModel::seeded(config.clone(), 17);
    let local_path = bin_temp_path("local_3d_continuation_ok.bpk");
    let local_manifest = BpkModelManifest::from_model(
        &model,
        grid.clone(),
        Some(format!(
            "ablation-rust:{UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE}"
        )),
    );
    crate::import::save_manifest(&local_path, &local_manifest).unwrap();

    let (_loaded, _grid, source) = load_conditionless_local_base_model(
        &local_path,
        MeshTargetArg::Torus,
        UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE,
    )
    .unwrap();
    let target_mismatch_err = load_conditionless_local_base_model(
        &local_path,
        MeshTargetArg::Teapot,
        TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE,
    )
    .unwrap_err()
    .to_string();
    std::fs::remove_file(&local_path).ok();
    assert!(source.contains("continued-from="));
    assert!(source.contains("conditionless-local"));
    assert!(!source.contains("position-field"));
    assert!(!source.contains("seed-frame"));
    assert!(!source.contains("render-proxy-rust"));
    assert!(
        target_mismatch_err.contains("target-mismatched lineage"),
        "continuation must not relabel a torus local base as a teapot local base"
    );

    let requested_source_mismatch = load_conditionless_local_base_model(
        &local_path,
        MeshTargetArg::Torus,
        TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE,
    )
    .unwrap_err()
    .to_string();
    assert!(
        requested_source_mismatch.contains("target source does not match target"),
        "continuation must not accept an output source for another mesh target"
    );

    let shortcut_path = bin_temp_path("local_3d_continuation_shortcut.bpk");
    let shortcut_manifest = BpkModelManifest::from_model(
        &model,
        grid.clone(),
        Some(format!(
            "ablation-rust:{UV_TORUS_POSITION_FIELD_TARGET_SOURCE}"
        )),
    );
    crate::import::save_manifest(&shortcut_path, &shortcut_manifest).unwrap();
    let shortcut_err = load_conditionless_local_base_model(
        &shortcut_path,
        MeshTargetArg::Torus,
        UV_TORUS_CONDITIONLESS_LOCAL_TARGET_SOURCE,
    )
    .unwrap_err();
    std::fs::remove_file(&shortcut_path).ok();
    assert!(shortcut_err.to_string().contains("shortcut lineage"));

    let mut position_config = config;
    position_config.position_features = true;
    let position_model = NpaModel::seeded(position_config, 19);
    let position_path = bin_temp_path("local_3d_continuation_position_features.bpk");
    let position_manifest = BpkModelManifest::from_model(
        &position_model,
        grid,
        Some(format!(
            "ablation-rust:{TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE}"
        )),
    );
    crate::import::save_manifest(&position_path, &position_manifest).unwrap();
    let position_err = load_conditionless_local_base_model(
        &position_path,
        MeshTargetArg::Teapot,
        TEAPOT_CONDITIONLESS_LOCAL_TARGET_SOURCE,
    )
    .unwrap_err();
    std::fs::remove_file(&position_path).ok();
    assert!(position_err.to_string().contains("position-feature"));
}
