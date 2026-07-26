use super::*;
use std::{
    collections::hash_map::DefaultHasher,
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
};

#[test]
fn performance_values_keep_stable_display_widths() {
    for (value, expected) in [
        (Some(9.94), "    9.9"),
        (Some(99.94), "   99.9"),
        (Some(999.94), "  999.9"),
        (Some(1_024.0), "  1.02k"),
        (Some(1_024_000.0), "  1.02M"),
        (None, "   --.-"),
    ] {
        let formatted = format_rate(value);
        assert_eq!(formatted, expected);
        assert_eq!(formatted.len(), 7);
    }
    assert_eq!(format_counter(9), "       9");
    assert_eq!(format_counter(99), "      99");
}

#[test]
fn performance_telemetry_publishes_one_coherent_snapshot() {
    let telemetry = AutomataPerformanceTelemetry::default();
    telemetry.publish(AutomataPerformanceSnapshot {
        render_thread_active: true,
        adaptive: true,
        completed_steps: 512,
        resident_particle_count: 3_070,
        dynamics_particle_count: 3_070,
        support_bin_count: 3,
        requested_support_bin_count: 3,
        min_material_radius: 0.01,
        median_material_radius: 0.01,
        max_material_radius: 0.02,
        split_events: 7,
        merge_events: 5,
    });
    let snapshot = telemetry.snapshot();
    assert!(snapshot.render_thread_active);
    assert!(snapshot.adaptive);
    assert_eq!(snapshot.completed_steps, 512);
    assert_eq!(snapshot.resident_particle_count, 3_070);
    assert_eq!(snapshot.support_bin_count, 3);
    assert_eq!(
        snapshot.max_material_radius / snapshot.min_material_radius,
        2.0
    );
    assert_eq!((snapshot.split_events, snapshot.merge_events), (7, 5));
}

#[test]
fn performance_label_system_uses_disjoint_text_queries() {
    let mut app = App::new();
    app.insert_resource(Time::<Real>::default())
        .insert_resource(AutomataRuntime::default())
        .init_resource::<AutomataPerformanceTelemetry>()
        .init_resource::<PerformanceUiState>()
        .add_systems(Update, update_performance_labels);
    let frame = app
        .world_mut()
        .spawn((Text::new(""), PerformanceFrameLabel))
        .id();
    let fps = app
        .world_mut()
        .spawn((Text::new(""), PerformanceFpsLabel))
        .id();
    let step_rate = app
        .world_mut()
        .spawn((Text::new(""), PerformanceStepRateLabel))
        .id();
    let adaptive = app
        .world_mut()
        .spawn((Text::new(""), AdaptiveDiagnosticsLabel))
        .id();

    app.update();

    assert_eq!(app.world().entity(frame).get::<Text>().unwrap().0.len(), 8);
    assert_eq!(app.world().entity(fps).get::<Text>().unwrap().0.len(), 7);
    assert_eq!(
        app.world().entity(step_rate).get::<Text>().unwrap().0.len(),
        7
    );
    assert!(app.world().entity(adaptive).get::<Text>().is_some());
}

#[test]
fn m_key_toggles_ui_visibility() {
    let mut app = App::new();
    app.init_resource::<AutomataUiState>()
        .insert_resource(ButtonInput::<KeyCode>::default())
        .add_systems(Update, toggle_ui_visibility);
    let root = app
        .world_mut()
        .spawn((AutomataUiRoot, Visibility::Inherited))
        .id();

    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .press(KeyCode::KeyM);
    app.update();

    assert!(!app.world().resource::<AutomataUiState>().visible);
    assert_eq!(
        *app.world().entity(root).get::<Visibility>().unwrap(),
        Visibility::Hidden
    );
}

#[test]
fn catalog_selection_preserves_visualization_settings() {
    let mut settings = AutomataSettings {
        render_scale: 1.5,
        render_opacity: 0.375,
        steps_per_frame: 3,
        training_learning_rate: 0.004,
        model_path: Some("previous-model.bpk".to_string()),
        ..Default::default()
    };
    let mut runtime = AutomataRuntime::default();

    select_catalog_entry(ModelCatalogKey::Growing3dGs, &mut settings, &mut runtime);

    assert_eq!(settings.preset, AutomataPreset::Growing3dGs);
    assert_eq!(settings.particle_count, 1024);
    assert_eq!(settings.steps_per_frame, 3);
    assert_eq!(settings.seed, RolloutConfig::default().seed);
    assert!((settings.reference_seed_scale - 0.35).abs() < f32::EPSILON);
    assert!((settings.render_scale - 1.5).abs() < f32::EPSILON);
    assert!((settings.render_opacity - 0.375).abs() < f32::EPSILON);
    assert!((settings.training_learning_rate - 0.004).abs() < f32::EPSILON);

    if catalog_entry_is_available(catalog_entry(ModelCatalogKey::UvTorusMorphogen3d)) {
        select_catalog_entry(
            ModelCatalogKey::UvTorusMorphogen3d,
            &mut settings,
            &mut runtime,
        );

        assert_eq!(settings.preset, AutomataPreset::Growing3dGs);
        assert_eq!(settings.particle_count, 1024);
        assert_eq!(settings.steps_per_frame, 3);
        assert_eq!(settings.seed, CATALOG_3D_GROWTH_SEED);
        assert_eq!(settings.seed_mode, ParticleSeed::TorusGrowth3d);
        assert!((settings.reference_seed_scale - 0.54).abs() < f32::EPSILON);
        assert!((settings.render_scale - 1.5).abs() < f32::EPSILON);
        assert!((settings.render_opacity - 0.375).abs() < f32::EPSILON);
        assert!((settings.training_learning_rate - 0.004).abs() < f32::EPSILON);
    }

    let teapot_available =
        catalog_entry_is_available(catalog_entry(ModelCatalogKey::TeapotMorphogen3d));
    if teapot_available {
        select_catalog_entry(
            ModelCatalogKey::TeapotMorphogen3d,
            &mut settings,
            &mut runtime,
        );

        assert_eq!(settings.preset, AutomataPreset::Growing3dGs);
        assert_eq!(settings.particle_count, 1024);
        assert_eq!(settings.steps_per_frame, 2);
        assert_eq!(settings.seed, CATALOG_3D_GROWTH_SEED);
        assert_eq!(settings.seed_mode, ParticleSeed::TeapotGrowth3d);
        assert!((settings.reference_seed_scale - 0.72).abs() < f32::EPSILON);
        assert!((settings.render_scale - 1.5).abs() < f32::EPSILON);
        assert!((settings.render_opacity - 0.375).abs() < f32::EPSILON);
        assert!((settings.training_learning_rate - 0.004).abs() < f32::EPSILON);
    }

    select_catalog_entry(ModelCatalogKey::Texture2d, &mut settings, &mut runtime);

    assert_eq!(settings.preset, AutomataPreset::Texture2d);
    assert_eq!(settings.particle_count, 4096);
    assert_eq!(
        settings.steps_per_frame,
        if teapot_available { 2 } else { 3 }
    );
    assert_eq!(settings.seed, RolloutConfig::default().seed);
    assert_eq!(settings.seed_mode, ParticleSeed::UniformCircle);
    assert!((settings.reference_seed_scale - 1.0).abs() < f32::EPSILON);
    assert!((settings.render_scale - 1.5).abs() < f32::EPSILON);
    assert!((settings.render_opacity - 0.375).abs() < f32::EPSILON);
    assert!((settings.training_learning_rate - 0.004).abs() < f32::EPSILON);
}

#[test]
fn catalog_keeps_only_latest_torus_regression_artifact() {
    let torus_entries = MODEL_CATALOG
        .iter()
        .filter(|entry| entry.title.contains("torus"))
        .collect::<Vec<_>>();
    assert_eq!(torus_entries.len(), 1);
    assert_eq!(torus_entries[0].key, ModelCatalogKey::UvTorusMorphogen3d);
    assert_eq!(
        catalog_seed_mode(torus_entries[0]),
        ParticleSeed::TorusGrowth3d
    );
    assert!(matches!(
        torus_entries[0].source,
        ModelCatalogSource::Bpk {
            primary: "assets/models/uv_torus_growth_3d.bpk",
            ..
        }
    ));
}

#[test]
fn visible_catalog_hides_blocked_3d_mesh_artifacts() {
    assert!(
        !VISIBLE_MODEL_CATALOG_KEYS.contains(&ModelCatalogKey::UvTorusMorphogen3d),
        "torus remains registered for regression loading but must not be selectable until validation passes"
    );
    assert!(
        !VISIBLE_MODEL_CATALOG_KEYS.contains(&ModelCatalogKey::TeapotMorphogen3d),
        "teapot remains registered for regression loading but must not be selectable until seed-varied robust validation passes"
    );
    assert!(VISIBLE_MODEL_CATALOG_KEYS.contains(&ModelCatalogKey::Growing3dGs));
}

#[test]
fn catalog_registers_teapot_as_blocked_growth_artifact() {
    let teapot_entries = MODEL_CATALOG
        .iter()
        .filter(|entry| entry.title.contains("teapot"))
        .collect::<Vec<_>>();
    assert_eq!(teapot_entries.len(), 1);
    assert_eq!(teapot_entries[0].key, ModelCatalogKey::TeapotMorphogen3d);
    assert_eq!(teapot_entries[0].particle_count, 1024);
    assert!(
        teapot_entries[0].kind.contains("hidden local regression"),
        "teapot should stay hidden until strict no-scaffold lineage and robust held-out seed validation pass"
    );
    assert_eq!(
        catalog_seed_mode(teapot_entries[0]),
        ParticleSeed::TeapotGrowth3d
    );
    assert!(matches!(
        teapot_entries[0].source,
        ModelCatalogSource::Bpk {
            primary: "assets/models/teapot_growth_3d.bpk",
            ..
        }
    ));
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[test]
fn catalog_3d_default_uses_auto_gpu_neighbor_mode() {
    let mut runtime = AutomataRuntime::default();
    let (config, hashgrid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    runtime.model = NpaModel::seeded(config, 42);
    runtime.hashgrid = hashgrid;

    let mut settings = AutomataSettings {
        preset: AutomataPreset::Growing3dGs,
        particle_count: 1024,
        gpu_neighbor_mode: WgpuNeighborMode::Auto,
        ..Default::default()
    };
    assert_eq!(
        effective_gpu_neighbor_mode(&runtime, &settings),
        WgpuNeighborMode::Auto
    );

    settings.particle_count = 4096;
    assert_eq!(
        effective_gpu_neighbor_mode(&runtime, &settings),
        WgpuNeighborMode::Auto
    );

    settings.gpu_neighbor_mode = WgpuNeighborMode::LinkedList;
    assert_eq!(
        effective_gpu_neighbor_mode(&runtime, &settings),
        WgpuNeighborMode::LinkedList
    );
}

#[test]
fn hidden_3d_bpk_entries_are_blocked_local_regression_artifacts() {
    for key in [
        ModelCatalogKey::UvTorusMorphogen3d,
        ModelCatalogKey::TeapotMorphogen3d,
    ] {
        let entry = catalog_entry(key);
        assert!(
            !VISIBLE_MODEL_CATALOG_KEYS.contains(&key),
            "{} must stay hidden until it is replaced by a strict-passing no-scaffold artifact",
            entry.title
        );
        assert!(
            entry.kind.contains("hidden local regression"),
            "{} should disclose why it is registered but not selectable",
            entry.title
        );
        let Some(path) = resolved_catalog_model_path(entry) else {
            continue;
        };
        let manifest = burn_automata::import::load_manifest(&path)
            .unwrap_or_else(|err| panic!("failed to load {path}: {err}"));

        assert_eq!(manifest.config.spatial_dims, 3, "{path}");
        assert!(
            !manifest.config.position_features,
            "{path} must not depend on absolute position features"
        );
        let source = manifest.source.as_deref().unwrap_or_default();
        let expected_source = match key {
            ModelCatalogKey::UvTorusMorphogen3d => {
                "render-refined-rust:ablation-rust:uv-torus-3d:conditionless-local-random-ball-rollout-ablation"
            }
            ModelCatalogKey::TeapotMorphogen3d => {
                "retimed-local-front:hidden=skipped:gain=2:alpha=1:front_retime=false:active_opacity_hidden=skipped:active_opacity_gain=skipped:opacity_bias=skipped:material_opacity_bias=0.55:base=render-refined-rust:ablation-rust:utah-teapot-2026:conditionless-local-random-ball-rollout-ablation"
            }
            _ => unreachable!("only 3D growth catalog entries are checked here"),
        };
        assert_eq!(
            source, expected_source,
            "{path} should point at the current reviewed latest dynamic 3D growth artifact"
        );
        assert!(
            (source.starts_with("render-refined-rust:")
                || source.starts_with("retimed-local-front:"))
                && source.contains("conditionless-local")
                && !source.contains("position-field")
                && !source.contains("seed-frame")
                && !source.contains("render-proxy-rust"),
            "{path} must use latest local render-refinement lineage without target-assigned shortcuts, source={source}"
        );
        let seed_mode = catalog_seed_mode(entry);
        assert!(matches!(
            seed_mode,
            ParticleSeed::TorusGrowth3d | ParticleSeed::TeapotGrowth3d
        ));
        assert!(
            burn_automata::rollout::growth_3d_seed_writes_coordinate_scaffold(seed_mode),
            "{path} should run with its source-matching random-ball diagnostic seed; strict catalog promotion must replace it with a no-scaffold source/seed lineage"
        );
    }
}

#[cfg(feature = "splatting")]
#[test]
fn automata_camera_viewport_centers_right_pane_when_ui_visible() {
    let viewport = automata_camera_viewport(UVec2::new(1600, 900), 1.0, true)
        .expect("wide window should allocate right-pane viewport");

    assert_eq!(viewport.physical_position, UVec2::new(540, 0));
    assert_eq!(viewport.physical_size, UVec2::new(1060, 900));
    assert!(automata_camera_viewport(UVec2::new(1600, 900), 1.0, false).is_none());
    assert!(automata_camera_viewport(UVec2::new(700, 900), 1.0, true).is_none());
}

#[cfg(feature = "splatting")]
#[test]
fn automata_camera_viewport_uses_physical_scale_factor() {
    let viewport = automata_camera_viewport(UVec2::new(3200, 1800), 2.0, true)
        .expect("hidpi window should allocate right-pane viewport");

    assert_eq!(viewport.physical_position, UVec2::new(1080, 0));
    assert_eq!(viewport.physical_size, UVec2::new(2120, 1800));
}

#[cfg(all(feature = "splatting", not(feature = "gpu_wgpu")))]
#[test]
fn cpu_trace_gaussian_fallback_writes_visible_gaussian() {
    let runtime = AutomataRuntime::default();
    let cfg = RolloutConfig {
        particle_count: 32,
        steps: 1,
        seed_scale: 0.2,
        update_prob: 1.0,
        ..RolloutConfig::default()
    };
    let trace = run_rollout(
        &runtime.model,
        &runtime.hashgrid,
        &cfg,
        ParticleSeed::UniformCircle,
    )
    .unwrap();
    let gaussian = trace_gaussian(&runtime, &trace, 0);
    assert_eq!(gaussian.position_visibility.visibility, 1.0);
    assert!(gaussian.scale_opacity.scale[0] > 0.0);
    assert!(gaussian.scale_opacity.opacity > 0.0);
    assert!(
        gaussian
            .spherical_harmonic
            .coefficients
            .iter()
            .all(|value| value.is_finite())
    );
}

#[cfg(feature = "splatting")]
#[test]
fn adaptive_gaussian_uses_material_footprint_as_dynamic_scale() {
    let radius = 0.0125_f32;
    let gaussian = adaptive_particle_gaussian(
        [0.1, -0.2, 0.0, 0.0],
        &[0.0, 0.1, -0.1],
        2,
        AdaptiveGaussianMaterial {
            represented_measure: std::f32::consts::PI * radius.powi(2),
            render_footprint: radius,
            display_scale_per_footprint: 1.0,
        },
        0.7,
    );
    assert_eq!(gaussian.position_visibility.visibility, 1.0);
    assert_eq!(gaussian.scale_opacity.scale, [0.0125; 3]);
    assert!((gaussian.scale_opacity.opacity - 0.7).abs() < 1.0e-6);
}

#[cfg(feature = "splatting")]
#[test]
fn adaptive_gaussian_is_isotropic_and_preserves_measure() {
    let represented_measure = 0.001_f32;
    let footprint = (represented_measure / std::f32::consts::PI).sqrt();
    let gaussian = adaptive_particle_gaussian(
        [0.0; 4],
        &[0.0; 3],
        2,
        AdaptiveGaussianMaterial {
            represented_measure,
            render_footprint: footprint,
            display_scale_per_footprint: 1.0,
        },
        1.0,
    );
    assert_eq!(gaussian.scale_opacity.scale, [footprint; 3]);
    assert_eq!(gaussian.rotation.rotation, [1.0, 0.0, 0.0, 0.0]);
    let reconstructed_measure = std::f32::consts::PI
        * gaussian.scale_opacity.scale[0]
        * gaussian.scale_opacity.scale[1]
        * gaussian.scale_opacity.opacity;
    assert!((reconstructed_measure - represented_measure).abs() < 1.0e-6);
}

#[cfg(feature = "splatting")]
#[test]
fn adaptive_latent_state_cannot_change_gaussian_geometry() {
    let radius = 0.01875_f32;
    let material = AdaptiveGaussianMaterial {
        represented_measure: std::f32::consts::PI * radius.powi(2),
        render_footprint: radius,
        display_scale_per_footprint: 2.5,
    };
    let baseline = adaptive_particle_gaussian([0.0; 4], &[0.0; 16], 2, material, 1.0);
    let arbitrary_state = adaptive_particle_gaussian(
        [0.0; 4],
        &[
            -1.0e6, 1.0e6, -10.0, 10.0, -3.0, 3.0, -2.0, 2.0, -1.0, 1.0, 0.5, -0.5, 0.25, -0.25,
            0.75, -0.75,
        ],
        2,
        material,
        1.0,
    );

    assert_eq!(
        arbitrary_state.scale_opacity.scale,
        baseline.scale_opacity.scale
    );
    assert_eq!(
        arbitrary_state.rotation.rotation,
        baseline.rotation.rotation
    );
    assert_eq!(
        arbitrary_state.scale_opacity.opacity,
        baseline.scale_opacity.opacity
    );
    assert_eq!(arbitrary_state.scale_opacity.scale, [radius * 2.5; 3]);
    assert_eq!(arbitrary_state.rotation.rotation, [1.0, 0.0, 0.0, 0.0]);
}

#[cfg(feature = "splatting")]
#[test]
fn adaptive_gaussian_preserves_multiscale_material_radius_ratio() {
    let gaussian_at = |radius: f32| {
        adaptive_particle_gaussian(
            [0.0; 4],
            &[0.0; 3],
            2,
            AdaptiveGaussianMaterial {
                represented_measure: std::f32::consts::PI * radius.powi(2),
                render_footprint: radius,
                display_scale_per_footprint: 3.0,
            },
            1.0,
        )
    };
    let fine = gaussian_at(0.00625);
    let coarse = gaussian_at(0.025);
    assert!((coarse.scale_opacity.scale[0] / fine.scale_opacity.scale[0] - 4.0).abs() < 1.0e-6);
    assert_eq!(fine.scale_opacity.opacity, 1.0);
    assert_eq!(coarse.scale_opacity.opacity, 1.0);
}

#[cfg(feature = "splatting")]
#[test]
fn adaptive_equal_resolution_matches_fixed_npa_viewer_scale() {
    let base = burn_automata::NpaModel::upstream_seeded(burn_automata::NpaConfig::growing_2d(), 7);
    let mut config = burn_automata::AdaptiveNpaConfig::growing_2d();
    config.base_rule_footprint = 0.003125;
    config.reference_footprint = 0.003125;
    let model = burn_automata::AdaptiveNpaModel::seeded(base, config, 9).unwrap();
    let radius = model.config.base_rule_footprint();
    let gaussian = adaptive_particle_gaussian(
        [0.0; 4],
        &[0.0; 3],
        2,
        AdaptiveGaussianMaterial {
            represented_measure: std::f32::consts::PI * radius.powi(2),
            render_footprint: radius,
            display_scale_per_footprint: adaptive_display_scale_per_footprint(&model),
        },
        1.0,
    );
    let fixed_scale = model.rule.config.eps0 * 0.12;
    assert!((gaussian.scale_opacity.scale[0] - fixed_scale).abs() < 1.0e-6);
}

#[test]
fn live_training_probe_updates_convergence_metrics_and_model_revision() {
    let mut runtime = AutomataRuntime::default();
    let cfg = RolloutConfig {
        particle_count: 32,
        steps: 1,
        seed_scale: 0.2,
        ..RolloutConfig::default()
    };
    let trace = run_rollout(
        &runtime.model,
        &runtime.hashgrid,
        &cfg,
        ParticleSeed::UniformCircle,
    )
    .unwrap();
    let previous_revision = runtime.model_revision;

    let settings = AutomataSettings::default();
    let hashgrid = effective_hashgrid(&runtime, &settings);
    update_training_probe(&mut runtime, &trace, &hashgrid, 1.0e-3);

    assert_eq!(runtime.training_step, 1);
    assert!(runtime.training_loss.is_some_and(f32::is_finite));
    assert!(runtime.training_grad_norm.is_some_and(f32::is_finite));
    assert_eq!(runtime.training_best_loss, runtime.training_loss);
    assert_ne!(runtime.model_revision, previous_revision);
}

#[test]
fn run_control_active_state_tracks_settings() {
    let mut settings = AutomataSettings::default();

    assert!(!run_control_is_active(RunControlKind::Pause, &settings));
    assert!(!run_control_is_active(RunControlKind::Train, &settings));

    settings.paused = true;
    settings.train_live = true;

    assert!(run_control_is_active(RunControlKind::Pause, &settings));
    assert!(run_control_is_active(RunControlKind::Train, &settings));
    assert!(!run_control_is_active(RunControlKind::Reset, &settings));
}

#[test]
fn control_probe_trace_is_bounded_and_finite() {
    let runtime = AutomataRuntime::default();
    let settings = AutomataSettings {
        particle_count: 4096,
        ..Default::default()
    };

    let trace = probe_trace_for_controls(&runtime, &settings, 64).unwrap();

    assert_eq!(trace.particle_count, 64);
    assert_eq!(trace.steps, 1);
    assert!(
        trace
            .positions
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
    assert!(trace.states.iter().all(|value| value.is_finite()));
}

#[test]
fn training_probe_interval_triggers_on_crossing() {
    assert!(!crossed_interval(58, 59, TRAINING_INTERVAL_FRAMES));
    assert!(crossed_interval(59, 60, TRAINING_INTERVAL_FRAMES));
    assert!(crossed_interval(56, 64, TRAINING_INTERVAL_FRAMES));
    assert!(!crossed_interval(60, 64, TRAINING_INTERVAL_FRAMES));
}

#[test]
fn model_catalog_has_unique_keys_and_paths() {
    let mut keys = HashSet::new();
    let mut paths = HashSet::new();
    for entry in MODEL_CATALOG {
        assert!(
            keys.insert(entry.key),
            "duplicate catalog key {:?}",
            entry.key
        );
        if let ModelCatalogSource::Bpk { primary, .. } = entry.source {
            assert!(
                paths.insert(primary),
                "duplicate catalog primary path {primary}"
            );
        }
    }
}

#[test]
fn catalog_thumbnails_are_embedded_decodable_and_distinct() {
    let mut hashes: HashMap<u64, ModelCatalogKey> = HashMap::new();
    for entry in MODEL_CATALOG {
        let bytes = catalog_thumbnail_png(entry.key);
        assert!(
            bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
            "catalog thumbnail for {:?} is not a PNG",
            entry.key
        );

        let image = catalog_thumbnail_image(entry.key);
        assert_eq!(image.texture_descriptor.size.width, 96);
        assert_eq!(image.texture_descriptor.size.height, 72);
        assert_eq!(
            image.texture_descriptor.format,
            TextureFormat::Rgba8UnormSrgb
        );

        let data = image
            .data
            .as_ref()
            .expect("decoded catalog thumbnail should keep CPU pixel data");
        assert_eq!(data.len(), 96 * 72 * 4);

        let mut min_rgb = [u8::MAX; 3];
        let mut max_rgb = [u8::MIN; 3];
        for pixel in data.chunks_exact(4) {
            for channel in 0..3 {
                min_rgb[channel] = min_rgb[channel].min(pixel[channel]);
                max_rgb[channel] = max_rgb[channel].max(pixel[channel]);
            }
        }
        let dynamic_range = (0..3)
            .map(|channel| max_rgb[channel] - min_rgb[channel])
            .max()
            .unwrap_or_default();
        assert!(
            dynamic_range > 24,
            "catalog thumbnail for {:?} looks blank: min={min_rgb:?} max={max_rgb:?}",
            entry.key
        );

        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        let hash = hasher.finish();
        let duplicate = hashes.insert(hash, entry.key);
        assert!(
            duplicate.is_none(),
            "catalog thumbnail for {:?} duplicates {:?}",
            entry.key,
            duplicate
        );
    }
    assert_eq!(hashes.len(), MODEL_CATALOG.len());
}

#[test]
fn uv_torus_preview_image_is_large_and_colored() {
    let image = catalog_preview_image(ModelCatalogKey::UvTorusMorphogen3d, 0.0);
    assert_eq!(image.texture_descriptor.size.width, 320);
    assert_eq!(image.texture_descriptor.size.height, 232);

    let data = image
        .data
        .as_ref()
        .expect("preview image should keep CPU pixel data");
    let mut min_rgb = [u8::MAX; 3];
    let mut max_rgb = [u8::MIN; 3];
    for pixel in data.chunks_exact(4) {
        for channel in 0..3 {
            min_rgb[channel] = min_rgb[channel].min(pixel[channel]);
            max_rgb[channel] = max_rgb[channel].max(pixel[channel]);
        }
    }

    assert!(max_rgb[0] - min_rgb[0] > 80, "weak red range");
    assert!(max_rgb[1] - min_rgb[1] > 80, "weak green range");
    assert!(max_rgb[2] - min_rgb[2] > 80, "weak blue range");
}

#[test]
fn teapot_preview_image_is_large_and_colored() {
    let image = catalog_preview_image(ModelCatalogKey::TeapotMorphogen3d, 0.0);
    assert_eq!(image.texture_descriptor.size.width, 320);
    assert_eq!(image.texture_descriptor.size.height, 232);

    let data = image
        .data
        .as_ref()
        .expect("preview image should keep CPU pixel data");
    let mut min_rgb = [u8::MAX; 3];
    let mut max_rgb = [u8::MIN; 3];
    for pixel in data.chunks_exact(4) {
        for channel in 0..3 {
            min_rgb[channel] = min_rgb[channel].min(pixel[channel]);
            max_rgb[channel] = max_rgb[channel].max(pixel[channel]);
        }
    }
    let dynamic_range = (0..3)
        .map(|channel| max_rgb[channel] - min_rgb[channel])
        .max()
        .unwrap_or_default();
    assert!(
        dynamic_range > 80,
        "teapot preview looks blank: min={min_rgb:?} max={max_rgb:?}"
    );
}

#[cfg(feature = "splatting")]
#[test]
fn sorted_entry_capacity_covers_resize_handoff_without_full_slider_floor() {
    assert_eq!(sorted_entry_capacity(0), SORTED_ENTRY_MIN_CAPACITY);
    assert_eq!(sorted_entry_capacity(128), SORTED_ENTRY_MIN_CAPACITY);
    assert_eq!(sorted_entry_capacity(4096), SORTED_ENTRY_MIN_CAPACITY);
    assert_eq!(sorted_entry_capacity(65_536), 65_536);
}

#[cfg(feature = "splatting")]
#[test]
fn automata_cloud_settings_use_display_rgb_color_space() {
    let settings = AutomataSettings::default();
    let cloud_settings = automata_cloud_settings(&settings, 2);

    assert_eq!(
        cloud_settings.color_space,
        GaussianColorSpace::SrgbRec709Display
    );
    assert_eq!(cloud_settings.sort_mode, SortMode::None);

    let cloud_settings_3d = automata_cloud_settings(&settings, 3);
    assert_eq!(cloud_settings_3d.sort_mode, SortMode::Radix);
    assert_eq!(
        cloud_settings_3d.radix_sort_depth_bits,
        RadixSortDepthBits::Bits32
    );
}

#[cfg(all(feature = "splatting", feature = "gpu_wgpu"))]
#[test]
fn live_weight_snapshots_do_not_reseed_the_gpu_rollout() {
    let settings = AutomataSettings::default();
    let (config, hashgrid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
    let model_before = NpaModel::upstream_seeded(config.clone(), 1);
    let mut model_after = model_before.clone();
    model_after.weights.w1[0] += 0.25;

    let before =
        automata_render_reinit_key(&model_before, &hashgrid, &settings, WgpuNeighborMode::Auto);
    let after =
        automata_render_reinit_key(&model_after, &hashgrid, &settings, WgpuNeighborMode::Auto);

    assert_eq!(before, after);
}
