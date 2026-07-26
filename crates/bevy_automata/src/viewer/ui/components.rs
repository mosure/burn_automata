use super::*;

#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct StatusLabel;

#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct PerformanceFrameLabel;

#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct PerformanceFpsLabel;

#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct PerformanceStepRateLabel;

#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct AdaptiveDiagnosticsLabel;

#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct SettingsLabel;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::viewer) enum AutomataSliderKind {
    #[default]
    ParticleLog2,
    StepsPerFrame,
    UpdateProb,
    DtLog2,
    RenderScaleLog2,
    RenderOpacityLog2,
    TrainingLearningRateLog2,
    TrainingRolloutResetInterval,
}

#[derive(Component, Clone, Copy, Debug, Default)]
pub(in crate::viewer) struct AutomataSlider(pub(in crate::viewer) AutomataSliderKind);

#[derive(Component, Clone, Copy, Debug, Default)]
pub(in crate::viewer) struct AutomataSliderValueLabel(pub(in crate::viewer) AutomataSliderKind);

#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct AutomataSliderThumb;

#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct AutomataSliderFill;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::viewer) enum RunControlKind {
    #[default]
    Pause,
    Reset,
    Train,
}

#[derive(Component, Clone, Copy, Debug, Default)]
pub(in crate::viewer) struct RunControlButton(pub(in crate::viewer) RunControlKind);

#[derive(Component, Clone, Copy, Debug, Default)]
#[cfg_attr(not(feature = "hyper_dino"), allow(dead_code))]
pub(in crate::viewer) struct RunControlButtonLabel(pub(in crate::viewer) RunControlKind);

#[cfg(feature = "hyper_dino")]
#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct HyperImageButton;

#[cfg(feature = "hyper_dino")]
#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct HyperInferenceButton;

#[cfg(feature = "hyper_dino")]
#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct HyperInferenceButtonLabel;

#[cfg(feature = "hyper_dino")]
#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct ImageTargetSummary;

#[cfg(feature = "hyper_dino")]
#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct ImageTargetPreview;

#[cfg(feature = "hyper_dino")]
#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct ImageTargetName;

#[cfg(feature = "hyper_dino")]
#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct ImageTargetProgress;

#[cfg(feature = "hyper_dino")]
#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct AdaptiveTrainingCheckbox;

#[cfg(feature = "hyper_dino")]
#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct AdaptiveTrainingCheckboxMark;

#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct AutomataUiPanel;

#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct AutomataUiRoot;

#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct AutomataUiScrollArea;

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::viewer) struct ModelCatalogCard(pub(in crate::viewer) ModelCatalogKey);

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::viewer) struct ModelCatalogThumbnail(pub(in crate::viewer) ModelCatalogKey);

#[derive(Component, Clone, Copy, Debug, Default)]
pub(in crate::viewer) struct ModelCatalogTextSize(pub(in crate::viewer) f32);

#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct CatalogPreviewRoot;

#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct CatalogPreviewTitle;

#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct CatalogPreviewDetail;

#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct CatalogPreviewImage;

#[derive(Resource, Clone, Debug, Default)]
pub(in crate::viewer) struct CatalogPreviewState {
    pub(in crate::viewer) open: bool,
    pub(in crate::viewer) key: Option<ModelCatalogKey>,
    pub(in crate::viewer) last_pressed_key: Option<ModelCatalogKey>,
    pub(in crate::viewer) last_press_time: f64,
}

#[derive(Resource, Clone, Debug, Default)]
pub(in crate::viewer) struct CatalogPreviewImageState {
    pub(in crate::viewer) handle: Option<Handle<Image>>,
    pub(in crate::viewer) key: Option<ModelCatalogKey>,
}

#[cfg(feature = "splatting")]
#[derive(Resource, Clone, Debug, Default)]
pub(in crate::viewer) struct AutomataUiInputCapture {
    pub(in crate::viewer) active: bool,
}

#[derive(Resource, Clone, Debug)]
pub(in crate::viewer) struct AutomataUiState {
    pub(in crate::viewer) visible: bool,
}

impl Default for AutomataUiState {
    fn default() -> Self {
        Self { visible: true }
    }
}

#[derive(Clone, Debug, Default)]
pub(in crate::viewer) struct AutomataPerformanceSnapshot {
    pub(in crate::viewer) render_thread_active: bool,
    pub(in crate::viewer) adaptive: bool,
    pub(in crate::viewer) completed_steps: usize,
    pub(in crate::viewer) resident_particle_count: usize,
    pub(in crate::viewer) dynamics_particle_count: usize,
    pub(in crate::viewer) support_bin_count: usize,
    pub(in crate::viewer) requested_support_bin_count: usize,
    pub(in crate::viewer) min_material_radius: f32,
    pub(in crate::viewer) median_material_radius: f32,
    pub(in crate::viewer) max_material_radius: f32,
    pub(in crate::viewer) split_events: usize,
    pub(in crate::viewer) merge_events: usize,
}

#[derive(Resource, Clone, Debug, Default)]
pub(in crate::viewer) struct AutomataPerformanceTelemetry(
    std::sync::Arc<std::sync::RwLock<AutomataPerformanceSnapshot>>,
);

impl AutomataPerformanceTelemetry {
    pub(in crate::viewer) fn snapshot(&self) -> AutomataPerformanceSnapshot {
        self.0
            .read()
            .map(|snapshot| snapshot.clone())
            .unwrap_or_default()
    }

    pub(in crate::viewer) fn publish(&self, snapshot: AutomataPerformanceSnapshot) {
        if let Ok(mut current) = self.0.write() {
            *current = snapshot;
        }
    }
}

#[derive(Resource, Clone, Debug)]
pub(in crate::viewer) struct PerformanceUiState {
    pub(in crate::viewer) initialized: bool,
    pub(in crate::viewer) last_sample_seconds: f64,
    pub(in crate::viewer) last_completed_steps: usize,
    pub(in crate::viewer) smoothed_fps: Option<f64>,
    pub(in crate::viewer) smoothed_step_rate: Option<f64>,
}

impl Default for PerformanceUiState {
    fn default() -> Self {
        Self {
            initialized: false,
            last_sample_seconds: 0.0,
            last_completed_steps: 0,
            smoothed_fps: None,
            smoothed_step_rate: None,
        }
    }
}
