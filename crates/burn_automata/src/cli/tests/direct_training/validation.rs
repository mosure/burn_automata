use super::*;

#[test]
fn growth_3d_validation_rejects_shortcut_lineage() {
    let config = NpaConfig::growing_3dgs();
    let grid = crate::kernels::HashGridConfig::growing_3dgs();
    let model =
        local_growth_student_model(config.clone(), 13, 0.0, LOCAL_GROWTH_EXPANSION_GAIN).unwrap();
    let path = bin_temp_path("shortcut_growth3d.bpk");
    let manifest = BpkModelManifest::from_model(
        &model,
        grid,
        Some("render-proxy-rust:Torus:field-baseline".to_string()),
    );
    crate::import::save_manifest(&path, &manifest).unwrap();

    let report = growth_3d_validation_report(
        &path,
        MeshTargetArg::Torus,
        growth_validation_test_config(ParticleSeed::TorusGrowth3d),
    )
    .unwrap();
    std::fs::remove_file(&path).ok();

    assert!(!report.local_conditionless_lineage);
    assert!(!report.gate_passed);
    assert!(!report.strict_passed);
}
