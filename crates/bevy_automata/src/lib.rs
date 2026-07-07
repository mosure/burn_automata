mod viewer;

pub use viewer::{AutomataRuntime, AutomataSettings, AutomataViewerPlugin, run, run_with_settings};

#[cfg(all(feature = "headless", feature = "splatting", feature = "gpu_wgpu"))]
pub use viewer::headless::{
    CaptureMetrics, HeadlessExportConfig, HeadlessExportRecord, HeadlessExportReport,
    run_headless_export,
};

#[cfg(all(feature = "gpu_wgpu", feature = "splatting"))]
pub use viewer::{
    AutomataRenderDiagnostics, automata_executor_from_render_device, gaussian_storage_buffer_refs,
};
