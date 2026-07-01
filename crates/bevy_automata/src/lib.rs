mod viewer;

pub use viewer::{AutomataRuntime, AutomataSettings, AutomataViewerPlugin, run};

#[cfg(all(feature = "gpu_wgpu", feature = "splatting"))]
pub use viewer::{
    AutomataRenderDiagnostics, automata_executor_from_render_device, gaussian_storage_buffer_refs,
};
