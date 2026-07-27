use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use burn_automata::TriangleMeshTarget;
#[cfg(not(target_arch = "wasm32"))]
use burn_automata::{
    Mesh3dTrainingProgress, Mesh3dTrainingReport, NpaModel, kernels::HashGridConfig,
};
use crossbeam_channel::{Receiver, Sender, unbounded};

use super::*;

mod dialog;
mod training;
mod ui;

pub(super) use dialog::{
    handle_npa_mesh_drop, handle_open_npa_mesh_dialog, poll_mesh_target_sources,
};
pub(super) use training::handle_toggle_mesh_target_training;
#[cfg(not(target_arch = "wasm32"))]
pub(super) use training::poll_mesh_target_training;
pub(super) use ui::{
    sync_mesh_target_summary, sync_mesh_training_button_label, update_mesh_button_styles,
};

#[derive(Message, Clone, Copy, Debug, Default)]
pub(super) struct OpenNpaMesh;

#[derive(Message, Clone, Copy, Debug, Default)]
pub(super) struct ToggleMeshTargetTraining;

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum MeshTargetTrainingPhase {
    #[default]
    Empty,
    Ready,
    Running,
    Stopping,
    Complete,
    Failed,
}

#[derive(Clone, Debug)]
pub(super) struct MeshTargetSource {
    pub file_name: String,
    pub bytes: Arc<Vec<u8>>,
    pub target: Arc<TriangleMeshTarget>,
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
#[derive(Clone, Debug)]
pub(super) struct MeshTarget {
    pub id: u64,
    pub source: MeshTargetSource,
}

#[derive(Resource, Debug, Default)]
pub(super) struct MeshTargetTrainingState {
    pub target: Option<MeshTarget>,
    pub phase: MeshTargetTrainingPhase,
    pub step: usize,
    pub total_steps: usize,
    pub loss: Option<f32>,
    pub best_loss: Option<f32>,
    pub grad_norm: Option<f32>,
    pub refresh: usize,
    pub refreshes: usize,
    pub policy_horizon: usize,
    pub error: Option<String>,
    pub last_rollout_reset_step: usize,
    next_target_id: u64,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    next_job_id: u64,
    active_job_id: Option<u64>,
    cancel: Option<Arc<AtomicBool>>,
}

impl MeshTargetTrainingState {
    pub fn has_target(&self) -> bool {
        self.target.is_some()
    }

    pub fn is_training(&self) -> bool {
        matches!(
            self.phase,
            MeshTargetTrainingPhase::Running | MeshTargetTrainingPhase::Stopping
        )
    }

    pub fn train_action_available(&self) -> bool {
        matches!(
            self.phase,
            MeshTargetTrainingPhase::Ready
                | MeshTargetTrainingPhase::Running
                | MeshTargetTrainingPhase::Complete
                | MeshTargetTrainingPhase::Failed
        )
    }

    pub fn train_action_label(&self) -> &'static str {
        match self.phase {
            MeshTargetTrainingPhase::Running => "stop",
            MeshTargetTrainingPhase::Stopping => "stopping",
            MeshTargetTrainingPhase::Complete => "continue",
            MeshTargetTrainingPhase::Ready
            | MeshTargetTrainingPhase::Failed
            | MeshTargetTrainingPhase::Empty => "train 3d",
        }
    }

    pub fn set_source(&mut self, source: MeshTargetSource) {
        self.stop_active_job();
        self.next_target_id = self.next_target_id.wrapping_add(1).max(1);
        self.target = Some(MeshTarget {
            id: self.next_target_id,
            source,
        });
        self.phase = MeshTargetTrainingPhase::Ready;
        self.reset_progress();
    }

    pub fn clear_target(&mut self) {
        self.stop_active_job();
        self.target = None;
        self.phase = MeshTargetTrainingPhase::Empty;
        self.reset_progress();
    }

    fn reset_progress(&mut self) {
        self.step = 0;
        self.total_steps = 0;
        self.loss = None;
        self.best_loss = None;
        self.grad_norm = None;
        self.refresh = 0;
        self.refreshes = 0;
        self.policy_horizon = 0;
        self.error = None;
        self.last_rollout_reset_step = 0;
    }

    fn stop_active_job(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            cancel.store(true, Ordering::Release);
        }
        self.active_job_id = None;
    }
}

#[derive(Resource)]
pub(super) struct MeshTargetDialogChannel {
    pub sender: Sender<Result<MeshTargetSource, String>>,
    pub receiver: Receiver<Result<MeshTargetSource, String>>,
}

impl Default for MeshTargetDialogChannel {
    fn default() -> Self {
        let (sender, receiver) = unbounded();
        Self { sender, receiver }
    }
}

#[derive(Resource)]
#[cfg(not(target_arch = "wasm32"))]
pub(super) struct MeshTargetTrainingChannel {
    pub sender: Sender<MeshTargetTrainingEvent>,
    pub receiver: Receiver<MeshTargetTrainingEvent>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for MeshTargetTrainingChannel {
    fn default() -> Self {
        let (sender, receiver) = unbounded();
        Self { sender, receiver }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) enum MeshTargetTrainingEvent {
    Progress {
        job_id: u64,
        target_id: u64,
        progress: Mesh3dTrainingProgress,
    },
    Finished {
        job_id: u64,
        target_id: u64,
        result: Result<MeshTargetTrainingCompletion, String>,
    },
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) struct MeshTargetTrainingCompletion {
    pub model: NpaModel,
    pub hashgrid: HashGridConfig,
    pub report: Mesh3dTrainingReport,
}
