pub(crate) fn resolve_full_coverage_adjoint(
    full_coverage_adjoint: bool,
    no_full_coverage_adjoint: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    if full_coverage_adjoint && no_full_coverage_adjoint {
        return Err(std::io::Error::other(
            "--full-coverage-adjoint and --no-full-coverage-adjoint are mutually exclusive",
        )
        .into());
    }
    Ok(full_coverage_adjoint || !no_full_coverage_adjoint)
}

pub(crate) fn resolve_direct_selection_seed_training(
    direct_selection_seed_training: bool,
    no_direct_selection_seed_training: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    if direct_selection_seed_training && no_direct_selection_seed_training {
        return Err(std::io::Error::other(
            "--direct-selection-seed-training and --no-direct-selection-seed-training are mutually exclusive",
        )
        .into());
    }
    Ok(direct_selection_seed_training || !no_direct_selection_seed_training)
}
