use super::*;

fn test_manifest(source: &str) -> BpkModelManifest {
    let model = NpaModel::seeded(NpaConfig::growing_3dgs(), 7);
    BpkModelManifest::from_model(
        &model,
        crate::kernels::HashGridConfig::growing_3dgs(),
        Some(source.to_string()),
    )
}

#[test]
fn rejected_catalog_render_training_candidate_preserves_catalog_output() {
    let manifest = test_manifest("ablation-rust:uv-torus-3d:conditionless-local-test");
    let root = bin_temp_path("rejected_catalog_render_training_candidate");
    let model_output = root.join("assets/models/rejected_candidate.bpk");

    let candidate_path = save_render_training_manifest_for_validation(
        &model_output,
        &manifest,
        MeshTargetArg::Torus,
    )
    .unwrap()
    .expect("catalog-shaped output should stage a validation candidate");
    assert!(candidate_path.exists());
    assert!(
        !model_output.exists(),
        "catalog output must not be written before promotion validation passes"
    );

    let error = finalize_render_training_manifest_promotion(
        &model_output,
        &manifest,
        Some(candidate_path.as_path()),
        Some(std::io::Error::other("strict gate failed").into()),
    )
    .unwrap_err();

    assert!(error.to_string().contains("strict gate failed"));
    assert!(
        !candidate_path.exists(),
        "rejected candidate should be cleaned after report generation"
    );
    assert!(
        !model_output.exists(),
        "rejected catalog candidate must not overwrite the catalog model"
    );
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn accepted_catalog_render_training_candidate_promotes_after_validation() {
    let manifest = test_manifest("ablation-rust:utah-teapot-2026:conditionless-local-test");
    let root = bin_temp_path("accepted_catalog_render_training_candidate");
    let model_output = root.join("assets/models/accepted_candidate.bpk");

    let candidate_path = save_render_training_manifest_for_validation(
        &model_output,
        &manifest,
        MeshTargetArg::Teapot,
    )
    .unwrap()
    .expect("catalog-shaped output should stage a validation candidate");
    assert!(candidate_path.exists());
    assert!(!model_output.exists());

    finalize_render_training_manifest_promotion(
        &model_output,
        &manifest,
        Some(candidate_path.as_path()),
        None,
    )
    .unwrap();

    assert!(!candidate_path.exists());
    assert!(
        model_output.exists(),
        "catalog output should be written only after promotion validation succeeds"
    );
    let promoted = crate::import::load_manifest(&model_output).unwrap();
    assert_eq!(promoted.source, manifest.source);
    std::fs::remove_dir_all(root).ok();
}

#[test]
fn catalog_promotion_summary_records_rejected_validation_evidence() {
    let summary = CliCatalogPromotionSummary::from_validation_and_training_result(
        true,
        2,
        Vec::new(),
        Some("strict gate failed".to_string()),
    );

    assert!(summary.requested);
    assert_eq!(summary.validation_count, 2);
    assert!(!summary.validation_passed);
    assert_eq!(
        summary.rejection_reason.as_deref(),
        Some("strict gate failed")
    );

    let skipped =
        CliCatalogPromotionSummary::from_validation_and_training_result(false, 0, Vec::new(), None);
    assert!(!skipped.requested);
    assert_eq!(skipped.validation_count, 0);
    assert!(!skipped.validation_passed);
    assert!(skipped.training_signal_passed);
    assert!(skipped.missing_train_signal_rounds.is_empty());
    assert!(skipped.rejection_reason.is_none());
}

#[test]
fn catalog_promotion_summary_rejects_missing_training_signal() {
    let summary = CliCatalogPromotionSummary::from_validation_and_training_result(
        true,
        2,
        vec![0, 3],
        Some("direct rollout training signal missing for rounds [0, 3]".to_string()),
    );

    assert!(summary.requested);
    assert_eq!(summary.validation_count, 2);
    assert!(!summary.validation_passed);
    assert!(!summary.training_signal_passed);
    assert_eq!(summary.missing_train_signal_rounds, vec![0, 3]);
    assert!(
        summary
            .rejection_reason
            .as_deref()
            .unwrap()
            .contains("training signal missing")
    );
}
