use super::direct_accumulators::DirectObjectiveAccumulators;
use super::direct_channels::DirectObjectiveChannels;
use super::direct_liveness::collect_direct_liveness_objectives;
use super::direct_material::{
    DirectMaterialObjectiveInputs, accumulate_direct_material_objectives,
};
use super::direct_motion::collect_direct_motion_objectives;
use super::terminal::terminal_liveness_state_diagnostics;
use super::*;

pub(crate) fn direct_rollout_objective_diagnostics(
    model: &NpaModel,
    target: &TriangleMeshTarget,
    trajectory: &[RenderTrajectorySnapshot],
    cfg: &RenderProxyTrainingConfig,
) -> Result<DirectRolloutObjectiveDiagnostics, Box<dyn std::error::Error>> {
    if trajectory.is_empty() {
        return Ok(DirectRolloutObjectiveDiagnostics::default());
    }

    let channels = DirectObjectiveChannels::new(model, cfg);
    let mut accumulators = DirectObjectiveAccumulators::default();

    for snapshot in trajectory {
        let particle_count = snapshot.positions.len();
        if particle_count == 0 {
            continue;
        }
        accumulators.add_rows(particle_count);

        let motion = collect_direct_motion_objectives(
            model,
            target,
            snapshot,
            cfg,
            &channels,
            &mut accumulators,
        )?;
        let liveness = collect_direct_liveness_objectives(
            model,
            target,
            snapshot,
            cfg,
            &channels,
            &motion,
            &mut accumulators,
        );
        accumulate_direct_material_objectives(
            model,
            target,
            snapshot,
            cfg,
            &channels,
            DirectMaterialObjectiveInputs {
                motion: &motion,
                liveness: &liveness,
            },
            &mut accumulators,
        );
    }

    let terminal_liveness_state =
        terminal_liveness_state_diagnostics(model, trajectory, cfg, channels.liveness_update_cap);

    Ok(accumulators.into_diagnostics(trajectory.len(), terminal_liveness_state))
}
