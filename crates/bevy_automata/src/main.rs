#[cfg(feature = "headless")]
use std::path::PathBuf;

#[cfg(feature = "headless")]
use burn_automata::{AutomataPreset, ParticleSeed};
#[cfg(feature = "headless")]
use clap::{Args, Parser, Subcommand, ValueEnum};

#[cfg(all(not(feature = "headless"), not(target_arch = "wasm32")))]
fn main() {
    bevy_automata::run();
}

#[cfg(all(not(feature = "headless"), target_arch = "wasm32"))]
fn main() {
    bevy_automata::run();
}

#[cfg(feature = "headless")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = AutomataCli::parse();
    match cli.command {
        Some(AutomataCommand::View(args)) => {
            bevy_automata::run_with_settings(args.into_settings()?);
            Ok(())
        }
        Some(AutomataCommand::Export(args)) => {
            let report = bevy_automata::run_headless_export((*args).into_config())?;
            println!(
                "wrote {} capture(s) to {}",
                report.captures.len(),
                report.output_dir.display()
            );
            println!("report {}", report.report_path.display());
            Ok(())
        }
        None => {
            bevy_automata::run();
            Ok(())
        }
    }
}

#[cfg(feature = "headless")]
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct AutomataCli {
    #[command(subcommand)]
    command: Option<AutomataCommand>,
}

#[cfg(feature = "headless")]
#[derive(Subcommand, Debug)]
enum AutomataCommand {
    /// Open the interactive viewer, optionally loading a BPK model.
    View(ViewArgs),
    /// Render rollout PNGs without opening a window.
    Export(Box<ExportArgs>),
}

#[cfg(feature = "headless")]
#[derive(Args, Debug)]
struct ViewArgs {
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    adaptive_model: Option<PathBuf>,
    #[arg(long)]
    no_adaptive_bandwidth: bool,
    #[arg(long)]
    no_adaptive_topology: bool,
    #[arg(long, default_value_t = 4096)]
    particles: usize,
    #[arg(long, value_enum, default_value_t = PresetArg::Growing2d)]
    preset: PresetArg,
    #[arg(long, value_enum, default_value_t = SeedModeArg::UniformCircle)]
    seed_mode: SeedModeArg,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    seed_scale: Option<f32>,
    #[arg(long, default_value_t = 0.5)]
    update_prob: f32,
    #[arg(long, default_value_t = 1.0)]
    dt: f32,
    #[arg(long, default_value_t = 0.5)]
    render_scale: f32,
    #[arg(long, default_value_t = 2.0)]
    render_opacity: f32,
    /// Visualize the leading particle-state principal components as RGB.
    #[arg(long)]
    pca: bool,
}

#[cfg(feature = "headless")]
impl ViewArgs {
    fn into_settings(self) -> Result<bevy_automata::AutomataSettings, Box<dyn std::error::Error>> {
        if self.particles == 0 {
            return Err(std::io::Error::other("--particles must be greater than zero").into());
        }
        if !(0.0..=1.0).contains(&self.update_prob) || !self.update_prob.is_finite() {
            return Err(std::io::Error::other("--update-prob must be finite and in [0, 1]").into());
        }
        if !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(std::io::Error::other("--dt must be finite and positive").into());
        }
        if self.model.is_some() && self.adaptive_model.is_some() {
            return Err(std::io::Error::other(
                "--model and --adaptive-model are mutually exclusive",
            )
            .into());
        }
        let preset: AutomataPreset = self.preset.into();
        let model_path = self
            .model
            .map(|path| {
                if !path.is_file() {
                    return Err(std::io::Error::other(format!(
                        "--model does not exist or is not a file: {}",
                        path.display()
                    )));
                }
                Ok(path.display().to_string())
            })
            .transpose()?;
        let adaptive_model_path = self
            .adaptive_model
            .map(|path| {
                if !path.is_file() {
                    return Err(std::io::Error::other(format!(
                        "--adaptive-model does not exist or is not a file: {}",
                        path.display()
                    )));
                }
                Ok(path.display().to_string())
            })
            .transpose()?;
        let mut settings = bevy_automata::AutomataSettings {
            preset,
            particle_count: self.particles,
            seed_mode: self.seed_mode.into(),
            seed: self.seed,
            seed_scale: self
                .seed_scale
                .unwrap_or_else(|| burn_automata::NpaConfig::seed_scale_for_preset(preset)),
            update_prob: self.update_prob,
            dt: self.dt,
            render_scale: self.render_scale,
            render_opacity: self.render_opacity,
            pca_visualization: self.pca,
            model_path,
            adaptive_model_path,
            adaptive_bandwidth_enabled: !self.no_adaptive_bandwidth,
            adaptive_topology_enabled: !self.no_adaptive_topology,
            ..bevy_automata::AutomataSettings::default()
        };
        settings.reference_seed_scale = settings.seed_scale;
        Ok(settings)
    }
}

#[cfg(feature = "headless")]
#[derive(Args, Debug)]
struct ExportArgs {
    #[arg(long, default_value = "target/bevy_automata_headless")]
    output_dir: PathBuf,
    #[arg(long, default_value = "automata")]
    output_prefix: String,
    #[arg(long, default_value_t = 512)]
    width: u32,
    #[arg(long, default_value_t = 512)]
    height: u32,
    #[arg(long, default_value_t = 4096)]
    particles: usize,
    #[arg(long, default_value_t = 128)]
    steps: usize,
    #[arg(long)]
    capture_every: Option<usize>,
    #[arg(long, value_delimiter = ',')]
    capture_steps: Vec<usize>,
    #[arg(long, default_value_t = 8)]
    warmup_frames: usize,
    #[arg(long, default_value_t = 1)]
    steps_per_frame: usize,
    #[arg(long, value_enum, default_value_t = PresetArg::Growing2d)]
    preset: PresetArg,
    #[arg(long, value_enum, default_value_t = SeedModeArg::UniformCircle)]
    seed_mode: SeedModeArg,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    adaptive_model: Option<PathBuf>,
    #[arg(long)]
    no_adaptive_bandwidth: bool,
    #[arg(long)]
    no_adaptive_topology: bool,
    #[arg(long)]
    hyper_image: Option<PathBuf>,
    #[arg(long)]
    hyper_base: Option<PathBuf>,
    #[arg(long)]
    hyper_model: Option<PathBuf>,
    #[arg(long)]
    dino_model: Option<PathBuf>,
    #[arg(long, default_value_t = 224)]
    dino_image_size: usize,
    #[arg(long, default_value_t = 14)]
    dino_patch_size: usize,
    /// Locally erase embedded mesh state before a 3D recovery export.
    #[arg(long)]
    mesh_damage_radius: Option<f32>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    seed_scale: Option<f32>,
    #[arg(long, default_value_t = 0.5)]
    update_prob: f32,
    #[arg(long, default_value_t = 1.0)]
    dt: f32,
    #[arg(long, default_value_t = 0.5)]
    render_scale: f32,
    #[arg(long, default_value_t = 2.0)]
    render_opacity: f32,
    /// Visualize the leading particle-state principal components as RGB.
    #[arg(long)]
    pca: bool,
}

#[cfg(feature = "headless")]
impl ExportArgs {
    fn into_config(self) -> bevy_automata::HeadlessExportConfig {
        bevy_automata::HeadlessExportConfig {
            output_dir: self.output_dir,
            output_prefix: self.output_prefix,
            width: self.width,
            height: self.height,
            particles: self.particles,
            steps: self.steps,
            capture_every: self.capture_every,
            capture_steps: self.capture_steps,
            warmup_frames: self.warmup_frames,
            steps_per_frame: self.steps_per_frame,
            preset: self.preset.into(),
            seed_mode: self.seed_mode.into(),
            model_path: self.model,
            adaptive_model_path: self.adaptive_model,
            adaptive_bandwidth_enabled: !self.no_adaptive_bandwidth,
            adaptive_topology_enabled: !self.no_adaptive_topology,
            hyper_image_path: self.hyper_image,
            hyper_base_model_path: self.hyper_base,
            hyper_model_path: self.hyper_model,
            dino_model_path: self.dino_model,
            dino_image_size: self.dino_image_size,
            dino_patch_size: self.dino_patch_size,
            mesh_damage_radius: self.mesh_damage_radius,
            seed: self.seed,
            seed_scale: self.seed_scale,
            update_prob: self.update_prob,
            dt: self.dt,
            render_scale: self.render_scale,
            render_opacity: self.render_opacity,
            pca_visualization: self.pca,
        }
    }
}

#[cfg(feature = "headless")]
#[derive(Clone, Copy, Debug, ValueEnum)]
enum PresetArg {
    Growing2d,
    Texture2d,
    Growing3dGs,
    PointMnist,
}

#[cfg(feature = "headless")]
impl From<PresetArg> for AutomataPreset {
    fn from(value: PresetArg) -> Self {
        match value {
            PresetArg::Growing2d => Self::Growing2d,
            PresetArg::Texture2d => Self::Texture2d,
            PresetArg::Growing3dGs => Self::Growing3dGs,
            PresetArg::PointMnist => Self::PointMnist,
        }
    }
}

#[cfg(feature = "headless")]
#[derive(Clone, Copy, Debug, ValueEnum)]
enum SeedModeArg {
    Gaussian,
    Uniform,
    UniformCircle,
    Growth3d,
    SubstrateGrowth3d,
    LocalGrowth3d,
    LocalSubstrateGrowth3d,
    UvTorus3d,
    UvTorusDense3d,
    TorusFieldDense3d,
    TeapotFieldDense3d,
    TorusGrowth3d,
    TeapotGrowth3d,
    TorusSubstrateGrowth3d,
    TeapotSubstrateGrowth3d,
    TorusLocalGrowth3d,
    TeapotLocalGrowth3d,
    TorusLocalSubstrateGrowth3d,
    TeapotLocalSubstrateGrowth3d,
    TorusMorphogenDense3d,
    TeapotMorphogenDense3d,
}

#[cfg(feature = "headless")]
impl From<SeedModeArg> for ParticleSeed {
    fn from(value: SeedModeArg) -> Self {
        match value {
            SeedModeArg::Gaussian => Self::Gaussian,
            SeedModeArg::Uniform => Self::Uniform,
            SeedModeArg::UniformCircle => Self::UniformCircle,
            SeedModeArg::Growth3d => Self::Growth3d,
            SeedModeArg::SubstrateGrowth3d => Self::SubstrateGrowth3d,
            SeedModeArg::LocalGrowth3d => Self::LocalGrowth3d,
            SeedModeArg::LocalSubstrateGrowth3d => Self::LocalSubstrateGrowth3d,
            SeedModeArg::UvTorus3d => Self::UvTorus3d,
            SeedModeArg::UvTorusDense3d => Self::UvTorusDense3d,
            SeedModeArg::TorusFieldDense3d => Self::TorusFieldDense3d,
            SeedModeArg::TeapotFieldDense3d => Self::TeapotFieldDense3d,
            SeedModeArg::TorusGrowth3d => Self::TorusGrowth3d,
            SeedModeArg::TeapotGrowth3d => Self::TeapotGrowth3d,
            SeedModeArg::TorusSubstrateGrowth3d => Self::TorusSubstrateGrowth3d,
            SeedModeArg::TeapotSubstrateGrowth3d => Self::TeapotSubstrateGrowth3d,
            SeedModeArg::TorusLocalGrowth3d => Self::TorusLocalGrowth3d,
            SeedModeArg::TeapotLocalGrowth3d => Self::TeapotLocalGrowth3d,
            SeedModeArg::TorusLocalSubstrateGrowth3d => Self::TorusLocalSubstrateGrowth3d,
            SeedModeArg::TeapotLocalSubstrateGrowth3d => Self::TeapotLocalSubstrateGrowth3d,
            SeedModeArg::TorusMorphogenDense3d => Self::TorusMorphogenDense3d,
            SeedModeArg::TeapotMorphogenDense3d => Self::TeapotMorphogenDense3d,
        }
    }
}
