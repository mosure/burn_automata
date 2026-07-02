use super::*;

#[derive(Component, Clone, Debug, Default)]
pub(in crate::viewer) struct StatusLabel;

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
    Backward,
    Train,
}

#[derive(Component, Clone, Copy, Debug, Default)]
pub(in crate::viewer) struct RunControlButton(pub(in crate::viewer) RunControlKind);

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
