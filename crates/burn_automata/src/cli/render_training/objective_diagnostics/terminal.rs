use super::accumulator::OutputGradientAccumulator;
use super::*;

pub(super) fn terminal_liveness_state_diagnostics(
    model: &NpaModel,
    trajectory: &[RenderTrajectorySnapshot],
    cfg: &RenderProxyTrainingConfig,
    max_adjoint: f32,
) -> OutputGradientAccumulator {
    let mut accumulator = OutputGradientAccumulator::default();
    let terminal_gain = direct_terminal_liveness_gain(cfg);
    if terminal_gain <= 0.0 || !terminal_gain.is_finite() {
        return accumulator;
    }
    let Some(snapshot) = trajectory.last() else {
        return accumulator;
    };
    if snapshot.positions.is_empty()
        || snapshot.states.len()
            < snapshot
                .positions
                .len()
                .saturating_mul(model.config.state_dims)
        || model.config.state_dims <= GROWTH_3D_LIVENESS_CHANNEL
    {
        return accumulator;
    }
    let mut state_adjoint = vec![0.0_f32; snapshot.states.len()];
    add_liveness_front_state_adjoint(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        terminal_gain,
        cfg.liveness_front_radius,
        1.0,
        max_adjoint,
        &mut state_adjoint,
    );
    add_temporal_activation_schedule_state_adjoint(
        &model.config,
        &snapshot.positions,
        &snapshot.states,
        terminal_gain,
        cfg.liveness_front_radius,
        1.0,
        max_adjoint,
        &mut state_adjoint,
    );
    for row in 0..snapshot.positions.len() {
        accumulator
            .add_value(state_adjoint[row * model.config.state_dims + GROWTH_3D_LIVENESS_CHANNEL]);
    }
    accumulator
}
