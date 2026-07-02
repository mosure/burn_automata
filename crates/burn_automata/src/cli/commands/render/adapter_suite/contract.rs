use crate::cli::prelude::*;

const MINIMUM_MANY_TARGET_COUNT: usize = 8;
const MINIMUM_MANY_NON_CORE_TARGET_COUNT: usize = 6;
const MINIMUM_MANY_SHARED_BASE_TARGET_COUNT: usize = 6;
const MINIMUM_MANY_HOLDOUT_TARGET_COUNT: usize = 2;

pub(super) fn adapter_suite_contract(
    target_set: MeshTargetSetArg,
    explicit_targets_requested: bool,
    targets: &[MeshTargetArg],
    shared_base_targets: &[MeshTargetArg],
    holdout_targets: &[MeshTargetArg],
    adapter_training_targets: &[MeshTargetArg],
) -> CliRenderAdapterSuiteContract {
    let many_object_default = target_set == MeshTargetSetArg::Many && !explicit_targets_requested;
    let core_target_count = targets
        .iter()
        .filter(|target| is_core_target(**target))
        .count();
    let non_core_target_count = targets.len().saturating_sub(core_target_count);

    let target_count_passed = if many_object_default {
        targets.len() >= MINIMUM_MANY_TARGET_COUNT
    } else {
        !targets.is_empty()
    };
    let non_core_target_count_passed = if many_object_default {
        non_core_target_count >= MINIMUM_MANY_NON_CORE_TARGET_COUNT
    } else {
        true
    };
    let shared_base_target_count_passed = if many_object_default {
        shared_base_targets.len() >= MINIMUM_MANY_SHARED_BASE_TARGET_COUNT
    } else {
        !shared_base_targets.is_empty() || !targets.is_empty()
    };
    let holdout_target_count_passed = if many_object_default {
        holdout_targets.len() >= MINIMUM_MANY_HOLDOUT_TARGET_COUNT
    } else {
        true
    };
    let adapters_cover_all_targets = targets
        .iter()
        .all(|target| adapter_training_targets.contains(target))
        && unique_target_count(adapter_training_targets) == targets.len();
    let contract_passed = target_count_passed
        && non_core_target_count_passed
        && shared_base_target_count_passed
        && holdout_target_count_passed
        && adapters_cover_all_targets;

    CliRenderAdapterSuiteContract {
        many_object_default,
        explicit_targets_requested,
        target_count: targets.len(),
        core_target_count,
        non_core_target_count,
        shared_base_target_count: shared_base_targets.len(),
        holdout_target_count: holdout_targets.len(),
        adapter_target_count: adapter_training_targets.len(),
        minimum_many_target_count: MINIMUM_MANY_TARGET_COUNT,
        minimum_many_non_core_target_count: MINIMUM_MANY_NON_CORE_TARGET_COUNT,
        minimum_many_shared_base_target_count: MINIMUM_MANY_SHARED_BASE_TARGET_COUNT,
        minimum_many_holdout_target_count: MINIMUM_MANY_HOLDOUT_TARGET_COUNT,
        target_count_passed,
        non_core_target_count_passed,
        shared_base_target_count_passed,
        holdout_target_count_passed,
        adapters_cover_all_targets,
        contract_passed,
    }
}

fn is_core_target(target: MeshTargetArg) -> bool {
    matches!(target, MeshTargetArg::Torus | MeshTargetArg::Teapot)
}

fn unique_target_count(targets: &[MeshTargetArg]) -> usize {
    let mut unique = Vec::new();
    for target in targets {
        if !unique.contains(target) {
            unique.push(*target);
        }
    }
    unique.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_many_contract_requires_non_core_target_bank() {
        let targets = mesh_target_set_targets(MeshTargetSetArg::Many);
        let holdouts = vec![
            MeshTargetArg::Ellipsoid,
            MeshTargetArg::Capsule,
            MeshTargetArg::Cross,
        ];
        let shared_base_targets = targets
            .iter()
            .copied()
            .filter(|target| !holdouts.contains(target))
            .collect::<Vec<_>>();

        let contract = adapter_suite_contract(
            MeshTargetSetArg::Many,
            false,
            &targets,
            &shared_base_targets,
            &holdouts,
            &targets,
        );

        assert!(contract.many_object_default);
        assert_eq!(contract.target_count, 12);
        assert_eq!(contract.core_target_count, 2);
        assert_eq!(contract.non_core_target_count, 10);
        assert_eq!(contract.shared_base_target_count, 9);
        assert_eq!(contract.holdout_target_count, 3);
        assert!(contract.adapters_cover_all_targets);
        assert!(contract.contract_passed);
    }

    #[test]
    fn default_many_contract_rejects_core_only_regression() {
        let targets = mesh_target_set_targets(MeshTargetSetArg::Core);
        let contract = adapter_suite_contract(
            MeshTargetSetArg::Many,
            false,
            &targets,
            &targets,
            &[],
            &targets,
        );

        assert!(contract.many_object_default);
        assert!(!contract.target_count_passed);
        assert!(!contract.non_core_target_count_passed);
        assert!(!contract.holdout_target_count_passed);
        assert!(!contract.contract_passed);
    }

    #[test]
    fn explicit_small_subset_is_allowed_for_diagnostics() {
        let targets = vec![MeshTargetArg::Torus, MeshTargetArg::Teapot];
        let contract = adapter_suite_contract(
            MeshTargetSetArg::Many,
            true,
            &targets,
            &targets,
            &[],
            &targets,
        );

        assert!(!contract.many_object_default);
        assert!(contract.target_count_passed);
        assert!(contract.non_core_target_count_passed);
        assert!(contract.holdout_target_count_passed);
        assert!(contract.contract_passed);
    }

    #[test]
    fn contract_rejects_missing_adapter_targets() {
        let targets = mesh_target_set_targets(MeshTargetSetArg::Many);
        let holdouts = vec![
            MeshTargetArg::Ellipsoid,
            MeshTargetArg::Capsule,
            MeshTargetArg::Cross,
        ];
        let shared_base_targets = targets
            .iter()
            .copied()
            .filter(|target| !holdouts.contains(target))
            .collect::<Vec<_>>();
        let adapter_targets = targets[..targets.len() - 1].to_vec();

        let contract = adapter_suite_contract(
            MeshTargetSetArg::Many,
            false,
            &targets,
            &shared_base_targets,
            &holdouts,
            &adapter_targets,
        );

        assert!(!contract.adapters_cover_all_targets);
        assert!(!contract.contract_passed);
    }
}
