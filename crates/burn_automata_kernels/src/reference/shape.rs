use super::*;

pub(crate) fn check_shapes(
    positions: &[[f32; 4]],
    states: &[f32],
    batch_size: usize,
    particle_count: usize,
    state_dims: usize,
) -> KernelResult<()> {
    let expected_positions = batch_size * particle_count;
    if positions.len() != expected_positions {
        return Err(KernelError::PositionShape {
            positions: positions.len(),
            expected: expected_positions,
        });
    }
    let expected_states = expected_positions * state_dims;
    if states.len() != expected_states {
        return Err(KernelError::StateShape {
            states: states.len(),
            expected: expected_states,
        });
    }
    Ok(())
}
