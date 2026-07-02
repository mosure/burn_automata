use super::*;

#[test]
fn manifest_roundtrips_seeded_model() {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let model = NpaModel::seeded(config, 99);
    let manifest = BpkModelManifest::from_model(&model, grid, Some("unit-test".to_string()));
    let path = temp_path("manifest_roundtrip.json");

    burn_automata::import::save_manifest(&path, &manifest).unwrap();
    let loaded = burn_automata::import::load_manifest(&path).unwrap();
    fs::remove_file(&path).ok();

    assert_eq!(loaded.format_version, 1);
    assert_eq!(loaded.model_kind, "npa");
    assert_eq!(loaded.source.as_deref(), Some("unit-test"));
    assert_eq!(loaded.config, manifest.config);
    loaded.into_model().validate().unwrap();
}
#[test]
fn bpk_manifest_roundtrips_and_rejects_corruption() {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing2d);
    let model = NpaModel::seeded(config, 123);
    let manifest = BpkModelManifest::from_model(&model, grid, Some("binary-test".to_string()));
    let path = temp_path("manifest_roundtrip.bpk");

    let digest = burn_automata::import::save_manifest(&path, &manifest)
        .unwrap()
        .expect("bpk saves return payload digest");
    assert_eq!(digest.len(), 64);

    let loaded = burn_automata::import::load_manifest(&path).unwrap();
    assert_eq!(loaded.config, manifest.config);
    assert_eq!(loaded.weights.w1, manifest.weights.w1);

    let mut bytes = fs::read(&path).unwrap();
    fs::remove_file(&path).ok();
    let last = bytes.last_mut().unwrap();
    *last = last.wrapping_add(1);
    let err = burn_automata::import::decode_bpk_manifest(&bytes).unwrap_err();
    assert!(err.to_string().contains("checksum"));
}

#[test]
fn adapter_manifest_roundtrips_and_materializes_base_model() {
    let (config, grid) = NpaConfig::for_preset(AutomataPreset::Growing3dGs);
    let model = NpaModel::seeded(config.clone(), 1234);
    let base_manifest = BpkModelManifest::from_model(&model, grid, Some("shared-base".to_string()));
    let adapter = NpaLowRankAdapter::seeded(&config, 2, 2.0, 55);
    let manifest = burn_automata::import::BpkAdapterManifest::from_adapter(
        &base_manifest,
        Some("base.bpk".to_string()),
        adapter.clone(),
        Some("adapter-test".to_string()),
    )
    .unwrap();
    let path = temp_path("adapter_manifest.adapter.json");

    burn_automata::import::save_adapter_manifest(&path, &manifest).unwrap();
    let loaded = burn_automata::import::load_adapter_manifest(&path).unwrap();
    fs::remove_file(&path).ok();

    loaded.validate(&base_manifest).unwrap();
    assert_eq!(loaded.model_kind, "npa-lora-adapter");
    assert_eq!(loaded.adapter_parameter_count(), adapter.parameter_count());
    let materialized = loaded.materialize(&base_manifest).unwrap();
    let direct = adapter.apply_to_model(&model).unwrap();
    assert_eq!(materialized.weights.w1, direct.weights.w1);
    assert_eq!(materialized.weights.b2, direct.weights.b2);
}

#[test]
fn pytorch_npa_checkpoint_imports_to_bpk() {
    let input = temp_path("fake_npa.pth");
    let output = temp_path("fake_npa.bpk");
    write_fake_pytorch_checkpoint(&input);

    let report = burn_automata::import::import_model(&input, &output).unwrap();
    let loaded = burn_automata::import::load_manifest(&output).unwrap();
    fs::remove_file(&input).ok();
    fs::remove_file(&output).ok();

    assert_eq!(report.container, "bpk");
    assert!(report.sha256.is_some());
    assert_eq!(loaded.config.spatial_dims, 2);
    assert_eq!(loaded.config.state_dims, 1);
    assert_eq!(loaded.config.hidden_dims, 2);
    assert_eq!(loaded.hashgrid.eps, 0.1);
    assert_eq!(burn_automata::import::parameter_count(&loaded), 23);
}
#[test]
fn burn_tensor_bridge_preserves_shapes() {
    type B = burn::backend::NdArray;
    let device = Default::default();

    let positions = [[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]];
    let tensor = burn_automata::burn_bridge::positions_to_tensor::<B>(&positions, &device);
    assert_eq!(tensor.shape().dims(), [2, 4]);

    let states = [0.0, 1.0, 2.0, 3.0];
    let tensor = burn_automata::burn_bridge::states_to_tensor::<B>(&states, 2, 2, &device);
    assert_eq!(tensor.shape().dims(), [2, 2]);
}
#[test]
fn burn_tensor_forward_matches_cpu_update_rows() {
    type B = burn::backend::NdArray;

    let config = NpaConfig {
        hidden_dims: 8,
        ..NpaConfig::growing_2d()
    };
    let model = NpaModel::seeded(config.clone(), 5);
    let rows = 3;
    let features = (0..rows * config.perception_dims())
        .map(|idx| (idx as f32 * 0.03).sin())
        .collect::<Vec<_>>();
    let expected = model.forward_update_from_features(&features).unwrap();

    let device = Default::default();
    let tensor = burn::tensor::Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(features, [rows, config.perception_dims()]),
        &device,
    );
    let actual = model
        .forward_update_tensor::<B>(tensor, &device)
        .unwrap()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();

    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-5);
    }
}
