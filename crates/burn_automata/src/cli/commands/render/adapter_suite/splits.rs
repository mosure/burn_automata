use crate::cli::prelude::*;

pub(super) fn default_adapter_suite_shared_base_cycles(
    has_base_model: bool,
    target_set: MeshTargetSetArg,
    target_count: usize,
) -> usize {
    if has_base_model {
        0
    } else if target_set == MeshTargetSetArg::Many || target_count > 4 {
        2
    } else {
        1
    }
}

pub(super) fn resolve_adapter_suite_targets(
    targets: Vec<MeshTargetArg>,
    target_set: MeshTargetSetArg,
) -> Result<Vec<MeshTargetArg>, Box<dyn std::error::Error>> {
    let resolved = if targets.is_empty() {
        mesh_target_set_targets(target_set)
    } else {
        unique_targets(targets)
    };
    if resolved.is_empty() {
        return Err(std::io::Error::other("adapter suite requires at least one target").into());
    }
    Ok(resolved)
}

pub(super) fn effective_adapter_suite_auto_holdout_stride(
    requested_stride: usize,
    explicit_targets_requested: bool,
    target_set: MeshTargetSetArg,
    manual_holdout_targets: &[MeshTargetArg],
    target_count: usize,
) -> usize {
    if requested_stride != 0
        || explicit_targets_requested
        || target_set != MeshTargetSetArg::Many
        || !manual_holdout_targets.is_empty()
        || target_count < 4
    {
        return requested_stride;
    }
    4
}

pub(super) fn adapter_suite_auto_holdout_targets(
    targets: &[MeshTargetArg],
    stride: usize,
    offset: usize,
) -> Result<Vec<MeshTargetArg>, Box<dyn std::error::Error>> {
    if stride == 0 {
        return Ok(Vec::new());
    }
    if stride < 2 {
        return Err(std::io::Error::other(
            "--auto-holdout-stride must be 0 or >= 2 so at least one target remains trainable",
        )
        .into());
    }
    let offset = offset % stride;
    Ok(targets
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, target)| (index % stride == offset).then_some(target))
        .collect())
}

pub(super) fn adapter_suite_holdout_targets(
    manual_holdout_targets: Vec<MeshTargetArg>,
    auto_holdout_targets: Vec<MeshTargetArg>,
) -> Vec<MeshTargetArg> {
    let mut holdouts = manual_holdout_targets;
    holdouts.extend(auto_holdout_targets);
    unique_targets(holdouts)
}

pub(super) fn validate_holdout_targets(
    targets: &[MeshTargetArg],
    holdout_targets: &[MeshTargetArg],
) -> Result<(), Box<dyn std::error::Error>> {
    for holdout_target in holdout_targets {
        if !targets.contains(holdout_target) {
            return Err(std::io::Error::other(format!(
                "holdout target {} is not part of the adapter suite targets",
                mesh_target_slug(*holdout_target)
            ))
            .into());
        }
    }
    Ok(())
}

pub(super) fn adapter_suite_split(
    target: MeshTargetArg,
    holdout_targets: &[MeshTargetArg],
) -> CliRenderAdapterSuiteSplit {
    if holdout_targets.contains(&target) {
        CliRenderAdapterSuiteSplit::HoldoutAdapterOnly
    } else {
        CliRenderAdapterSuiteSplit::SharedBaseTrain
    }
}

pub(super) fn suite_report_shared_base_target_count(
    entries: &[CliRenderAdapterSuiteEntry],
) -> usize {
    entries
        .iter()
        .filter(|entry| entry.split == CliRenderAdapterSuiteSplit::SharedBaseTrain)
        .count()
}

pub(super) fn suite_report_holdout_target_count(entries: &[CliRenderAdapterSuiteEntry]) -> usize {
    entries
        .iter()
        .filter(|entry| entry.split == CliRenderAdapterSuiteSplit::HoldoutAdapterOnly)
        .count()
}

fn unique_targets(targets: Vec<MeshTargetArg>) -> Vec<MeshTargetArg> {
    let mut unique = Vec::with_capacity(targets.len());
    for target in targets {
        if !unique.contains(&target) {
            unique.push(target);
        }
    }
    unique
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_arg_adapter_suite_resolves_many_object_bank() {
        let targets = resolve_adapter_suite_targets(Vec::new(), MeshTargetSetArg::Many).unwrap();
        assert_eq!(targets, mesh_target_set_targets(MeshTargetSetArg::Many));
        assert!(targets.contains(&MeshTargetArg::Torus));
        assert!(targets.contains(&MeshTargetArg::Teapot));
        assert!(targets.contains(&MeshTargetArg::Capsule));
        assert!(targets.contains(&MeshTargetArg::Cross));
    }

    #[test]
    fn explicit_adapter_suite_targets_remain_focused_subset() {
        let targets = resolve_adapter_suite_targets(
            vec![MeshTargetArg::Sphere, MeshTargetArg::Cube],
            MeshTargetSetArg::Many,
        )
        .unwrap();
        assert_eq!(targets, vec![MeshTargetArg::Sphere, MeshTargetArg::Cube]);
    }

    #[test]
    fn many_object_adapter_suite_defaults_to_holdout_split() {
        let targets = mesh_target_set_targets(MeshTargetSetArg::Many);
        let stride = effective_adapter_suite_auto_holdout_stride(
            0,
            false,
            MeshTargetSetArg::Many,
            &[],
            targets.len(),
        );
        assert_eq!(stride, 4);
        let holdouts = adapter_suite_auto_holdout_targets(&targets, stride, 3).unwrap();
        assert_eq!(
            holdouts,
            vec![
                MeshTargetArg::Ellipsoid,
                MeshTargetArg::Capsule,
                MeshTargetArg::Cross
            ]
        );
        assert!(!holdouts.contains(&MeshTargetArg::Torus));
        assert!(!holdouts.contains(&MeshTargetArg::Teapot));
    }
}
