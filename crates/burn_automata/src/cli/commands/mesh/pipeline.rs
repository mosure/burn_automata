use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::cli::prelude::*;
use crate::{Mesh3dQualityReport, evaluate_mesh3d_model};

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct Mesh3dExperiment {
    mesh: Mesh3dInput,
    training: Mesh3dTrainingConfig,
    output: Mesh3dOutput,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
struct Mesh3dInput {
    builtin: Option<String>,
    path: Option<PathBuf>,
}

impl Default for Mesh3dInput {
    fn default() -> Self {
        Self {
            builtin: Some("utah_teapot".to_string()),
            path: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
struct Mesh3dOutput {
    model: PathBuf,
    report: PathBuf,
}

impl Default for Mesh3dOutput {
    fn default() -> Self {
        Self {
            model: PathBuf::from("artifacts/mesh3d/utah_teapot/model.bpk"),
            report: PathBuf::from("artifacts/mesh3d/utah_teapot/report.json"),
        }
    }
}

#[derive(Serialize)]
struct Mesh3dExperimentReport {
    target: Mesh3dTargetReport,
    training: Mesh3dTrainingReport,
    model_output: String,
}

#[derive(Serialize)]
struct Mesh3dTargetReport {
    source: String,
    vertices: usize,
    faces: usize,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
}

#[derive(Serialize)]
struct Mesh3dEvaluationReport {
    target: Mesh3dTargetReport,
    model: String,
    quality: Mesh3dQualityReport,
}

#[cfg(feature = "backend_wgpu")]
pub(crate) fn run_train_mesh_3d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::TrainMesh3d { config } = command else {
        unreachable!("run_train_mesh_3d called with the wrong command variant");
    };
    let experiment: Mesh3dExperiment = toml::from_str(&std::fs::read_to_string(&config)?)?;
    let (target, source) = load_target(&experiment.mesh, experiment.training.scale)?;
    let (bounds_min, bounds_max) = target.bounds();
    eprintln!(
        "mesh3d target={} vertices={} faces={} rows={} steps={} particles={}x{}",
        source,
        target.vertices.len(),
        target.faces.len(),
        experiment.training.dataset_particles * experiment.training.dataset_trajectories,
        experiment.training.steps,
        experiment.training.dataset_particles,
        experiment.training.dataset_trajectories,
    );
    let (model, hashgrid, training) = train_mesh3d_wgpu(&target, experiment.training.clone())?;
    ensure_parent(&experiment.output.model)?;
    ensure_parent(&experiment.output.report)?;
    let initialization = mesh3d_surface_initialization(
        &target,
        &model.config,
        experiment.training.evaluation.particle_count,
        experiment.training.seed,
    )?;
    let manifest =
        BpkModelManifest::from_model(&model, hashgrid, Some(format!("mesh3d-wgpu:{source}")))
            .with_initialization(initialization)?;
    crate::import::save_manifest(&experiment.output.model, &manifest)?;
    let report = Mesh3dExperimentReport {
        target: Mesh3dTargetReport {
            source,
            vertices: target.vertices.len(),
            faces: target.faces.len(),
            bounds_min,
            bounds_max,
        },
        training,
        model_output: experiment.output.model.display().to_string(),
    };
    std::fs::write(
        &experiment.output.report,
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "wrote {} and {} quality={} surface={:.5} density_psnr={:.2}dB color_psnr={:.2}dB",
        experiment.output.model.display(),
        experiment.output.report.display(),
        if report.training.quality.passed {
            "passed"
        } else {
            "failed"
        },
        report
            .training
            .quality
            .rollouts
            .last()
            .map_or(f32::NAN, |row| row.mean_surface_distance),
        report
            .training
            .quality
            .rollouts
            .last()
            .map_or(f32::NAN, |row| row.density_psnr_db),
        report
            .training
            .quality
            .rollouts
            .last()
            .map_or(f32::NAN, |row| row.color_psnr_db),
    );
    if !report.training.quality.passed {
        return Err(std::io::Error::other(format!(
            "mesh3d quality gates failed; see {}",
            experiment.output.report.display()
        ))
        .into());
    }
    Ok(())
}

#[cfg(not(feature = "backend_wgpu"))]
pub(crate) fn run_train_mesh_3d(_command: Command) -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("mesh3d training requires the backend_wgpu feature").into())
}

pub(crate) fn run_evaluate_mesh_3d(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    let Command::EvaluateMesh3d {
        config,
        model,
        report,
    } = command
    else {
        unreachable!("run_evaluate_mesh_3d called with the wrong command variant");
    };
    let experiment: Mesh3dExperiment = toml::from_str(&std::fs::read_to_string(&config)?)?;
    let (target, source) = load_target(&experiment.mesh, experiment.training.scale)?;
    let manifest = crate::import::load_manifest(&model)?;
    let hashgrid = manifest.hashgrid.clone();
    let model_value = manifest.into_model();
    let quality = evaluate_mesh3d_model(
        &model_value,
        &hashgrid,
        &target,
        &experiment.training.evaluation,
    )?;
    let (bounds_min, bounds_max) = target.bounds();
    ensure_parent(&report)?;
    let output = Mesh3dEvaluationReport {
        target: Mesh3dTargetReport {
            source,
            vertices: target.vertices.len(),
            faces: target.faces.len(),
            bounds_min,
            bounds_max,
        },
        model: model.display().to_string(),
        quality,
    };
    std::fs::write(&report, serde_json::to_vec_pretty(&output)?)?;
    let required = output
        .quality
        .rollouts
        .iter()
        .filter(|row| row.required_for_quality)
        .collect::<Vec<_>>();
    let worst_density = required
        .iter()
        .map(|row| row.density_psnr_db)
        .reduce(f32::min)
        .unwrap_or(f32::NAN);
    let worst_surface = required
        .iter()
        .map(|row| row.mean_surface_distance)
        .reduce(f32::max)
        .unwrap_or(f32::NAN);
    println!(
        "wrote {} quality={} worst_surface={worst_surface:.5} worst_density_psnr={worst_density:.2}dB",
        report.display(),
        if output.quality.passed {
            "passed"
        } else {
            "failed"
        },
    );
    if !output.quality.passed {
        return Err(std::io::Error::other(format!(
            "mesh3d quality gates failed; see {}",
            report.display()
        ))
        .into());
    }
    Ok(())
}

fn load_target(
    input: &Mesh3dInput,
    scale: f32,
) -> Result<(TriangleMeshTarget, String), Box<dyn std::error::Error>> {
    match (&input.path, input.builtin.as_deref()) {
        (Some(path), None) => {
            let source = path.display().to_string();
            let target = TriangleMeshTarget::from_obj_str(&std::fs::read_to_string(path)?, scale)?;
            Ok((target, source))
        }
        (None, Some("utah_teapot" | "teapot")) => Ok((
            TriangleMeshTarget::utah_teapot(scale)?,
            "builtin:utah_teapot".to_string(),
        )),
        (Some(_), Some(_)) => Err(std::io::Error::other(
            "mesh3d config must set exactly one of mesh.path or mesh.builtin",
        )
        .into()),
        (None, Some(value)) => {
            Err(std::io::Error::other(format!("unsupported mesh3d builtin {value:?}")).into())
        }
        (None, None) => {
            Err(std::io::Error::other("mesh3d config must set mesh.path or mesh.builtin").into())
        }
    }
}

fn ensure_parent(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}
