use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    AdamWConfig, NpaConfig, NpaModel, NpaWeights, SupervisedGradients, Target2dLossConfig,
    TargetImage2d, kernels::HashGridConfig, target_2d_upstream_one_step_with_gradients,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Npa2dTrainingParityConfig {
    #[serde(default)]
    pub(super) required: bool,
    #[serde(default = "default_feature_rmse")]
    max_feature_rmse: f64,
    #[serde(default = "default_forward_rmse")]
    max_forward_rmse: f64,
    #[serde(default = "default_loss_abs")]
    max_loss_abs: f64,
    #[serde(default = "default_gradient_cosine")]
    min_raw_gradient_cosine: f64,
    #[serde(default = "default_normalized_gradient_rmse")]
    max_normalized_gradient_rmse: f64,
    #[serde(default = "default_optimizer_rmse")]
    max_optimizer_rmse: f64,
}

#[derive(Debug, Serialize)]
pub(super) struct Npa2dTrainingParityReport {
    pub(super) present: bool,
    pub(super) required: bool,
    pub(super) passed: bool,
    pub(super) architecture: Option<String>,
    pub(super) batch_size: Option<usize>,
    pub(super) particle_count: Option<usize>,
    pub(super) comparisons: BTreeMap<String, TensorComparison>,
    pub(super) loss_differences: BTreeMap<String, f64>,
    pub(super) failures: Vec<String>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(super) struct TensorComparison {
    elements: usize,
    max_abs: f64,
    rmse: f64,
    relative_l2: f64,
    cosine: f64,
}

#[derive(Debug, Deserialize)]
struct RootFixture {
    training_step: Option<TrainingStepFixture>,
}

#[derive(Debug, Deserialize)]
struct TrainingStepFixture {
    batch_size: usize,
    particle_count: usize,
    spatial_dims: usize,
    state_dims: usize,
    hidden_dims: usize,
    perception_dims: usize,
    update_dims: usize,
    architecture: String,
    positions: Vec<f32>,
    states: Vec<f32>,
    update_mask: Vec<f32>,
    model: ParameterFixture,
    forward: ForwardFixture,
    loss: LossFixture,
    raw_gradients: ParameterFixture,
    normalized_gradients: ParameterFixture,
    optimizer: OptimizerFixture,
    updated_model: ParameterFixture,
}

#[derive(Debug, Deserialize)]
struct ParameterFixture {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct ForwardFixture {
    features: Vec<f32>,
    raw_update: Vec<f32>,
    dx: Vec<f32>,
    ds: Vec<f32>,
    next_positions: Vec<f32>,
    next_states: Vec<f32>,
    mean_dx_norm: f32,
}

#[derive(Debug, Deserialize)]
struct LossFixture {
    total: f32,
    components: BTreeMap<String, f32>,
    terms: BTreeMap<String, f32>,
}

#[derive(Debug, Deserialize)]
struct OptimizerFixture {
    name: String,
    learning_rate: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

pub(super) fn validate_training_step(
    fixture_path: &Path,
    target: &TargetImage2d,
    grid: &HashGridConfig,
    loss_config: Target2dLossConfig,
    config: &Npa2dTrainingParityConfig,
) -> Result<Npa2dTrainingParityReport, Box<dyn std::error::Error>> {
    let root: RootFixture = serde_json::from_str(&std::fs::read_to_string(fixture_path)?)?;
    let Some(fixture) = root.training_step else {
        let failures = if config.required {
            vec![format!(
                "required upstream training_step is absent from {}; regenerate with scripts/reference/selforg/export_selforg_npa_fixture.py --training-step",
                fixture_path.display()
            )]
        } else {
            Vec::new()
        };
        return Ok(Npa2dTrainingParityReport {
            present: false,
            required: config.required,
            passed: failures.is_empty(),
            architecture: None,
            batch_size: None,
            particle_count: None,
            comparisons: BTreeMap::new(),
            loss_differences: BTreeMap::new(),
            failures,
        });
    };

    validate_fixture_shape(&fixture)?;
    let model = NpaModel {
        config: NpaConfig::growing_2d(),
        weights: NpaWeights {
            w1: fixture.model.w1.clone(),
            b1: fixture.model.b1.clone(),
            w2: fixture.model.w2.clone(),
            b2: vec![0.0; fixture.update_dims],
        },
    };
    let positions = fixture
        .positions
        .chunks_exact(2)
        .map(|value| [value[0], value[1], 0.0, 0.0])
        .collect::<Vec<_>>();
    let output = target_2d_upstream_one_step_with_gradients(
        &model,
        grid,
        target,
        positions,
        fixture.states.clone(),
        fixture.update_mask.clone(),
        loss_config,
        AdamWConfig {
            learning_rate: fixture.optimizer.learning_rate,
            weight_decay: fixture.optimizer.weight_decay,
            grad_clip_norm: 0.0,
            beta1: fixture.optimizer.beta1,
            beta2: fixture.optimizer.beta2,
            epsilon: fixture.optimizer.epsilon,
        },
    )?;

    let mut comparisons = BTreeMap::new();
    insert_comparison(
        &mut comparisons,
        "forward.features",
        &output.features,
        &fixture.forward.features,
    )?;
    insert_comparison(
        &mut comparisons,
        "forward.raw_update",
        &output.raw_update,
        &fixture.forward.raw_update,
    )?;
    insert_comparison(
        &mut comparisons,
        "forward.dx",
        &flatten_spatial(&output.dx, 2),
        &fixture.forward.dx,
    )?;
    insert_comparison(
        &mut comparisons,
        "forward.ds",
        &output.ds,
        &fixture.forward.ds,
    )?;
    insert_comparison(
        &mut comparisons,
        "forward.next_positions",
        &flatten_spatial(&output.next_positions, 2),
        &fixture.forward.next_positions,
    )?;
    insert_comparison(
        &mut comparisons,
        "forward.next_states",
        &output.next_states,
        &fixture.forward.next_states,
    )?;
    insert_gradient_comparisons(
        &mut comparisons,
        "gradient.raw",
        &output.raw_gradients,
        &fixture.raw_gradients,
    )?;
    insert_gradient_comparisons(
        &mut comparisons,
        "gradient.normalized",
        &output.normalized_gradients,
        &fixture.normalized_gradients,
    )?;
    insert_parameter_comparisons(
        &mut comparisons,
        "optimizer.updated",
        &output.updated_model.weights,
        &fixture.updated_model,
    )?;

    let mut loss_differences = BTreeMap::new();
    insert_loss_difference(
        &mut loss_differences,
        "total",
        output.loss.total_loss,
        fixture.loss.total,
    );
    insert_loss_difference(
        &mut loss_differences,
        "splat_loss",
        output.loss.splat_loss,
        fixture
            .loss
            .components
            .get("splat_loss")
            .copied()
            .unwrap_or(f32::NAN),
    );
    insert_loss_difference(
        &mut loss_differences,
        "color_loss",
        output.loss.color_loss,
        fixture
            .loss
            .terms
            .get("splat_loss.color_loss")
            .copied()
            .unwrap_or(f32::NAN),
    );
    insert_loss_difference(
        &mut loss_differences,
        "density_loss",
        output.loss.density_loss,
        fixture
            .loss
            .terms
            .get("splat_loss.density_loss")
            .copied()
            .unwrap_or(f32::NAN),
    );
    insert_loss_difference(
        &mut loss_differences,
        "displacement_regularizer",
        output.loss.displacement_regularizer,
        fixture
            .loss
            .components
            .get("displacement_regularizer")
            .copied()
            .unwrap_or(f32::NAN),
    );
    insert_loss_difference(
        &mut loss_differences,
        "overflow_regularizer",
        output.loss.overflow_regularizer,
        fixture
            .loss
            .components
            .get("overflow_regularizer")
            .copied()
            .unwrap_or(f32::NAN),
    );
    insert_loss_difference(
        &mut loss_differences,
        "bound_regularizer",
        output.loss.bound_regularizer,
        fixture
            .loss
            .components
            .get("bound_regularizer")
            .copied()
            .unwrap_or(f32::NAN),
    );
    insert_loss_difference(
        &mut loss_differences,
        "mean_dx_norm",
        output.mean_dx_norm,
        fixture.forward.mean_dx_norm,
    );

    let mut failures = Vec::new();
    require_rmse(
        &comparisons,
        "forward.features",
        config.max_feature_rmse,
        &mut failures,
    );
    for name in [
        "forward.raw_update",
        "forward.dx",
        "forward.ds",
        "forward.next_positions",
        "forward.next_states",
    ] {
        require_rmse(&comparisons, name, config.max_forward_rmse, &mut failures);
    }
    for (name, difference) in &loss_differences {
        if !difference.is_finite() || *difference > config.max_loss_abs {
            failures.push(format!(
                "training parity {name} abs diff {difference:.6e} exceeds {:.6e}",
                config.max_loss_abs
            ));
        }
    }
    for name in ["gradient.raw.w1", "gradient.raw.b1", "gradient.raw.w2"] {
        let comparison = &comparisons[name];
        if !comparison.cosine.is_finite() || comparison.cosine < config.min_raw_gradient_cosine {
            failures.push(format!(
                "training parity {name} cosine {:.8} is below {:.8}",
                comparison.cosine, config.min_raw_gradient_cosine
            ));
        }
    }
    for name in [
        "gradient.normalized.w1",
        "gradient.normalized.b1",
        "gradient.normalized.w2",
    ] {
        require_rmse(
            &comparisons,
            name,
            config.max_normalized_gradient_rmse,
            &mut failures,
        );
    }
    for name in [
        "optimizer.updated.w1",
        "optimizer.updated.b1",
        "optimizer.updated.w2",
    ] {
        require_rmse(&comparisons, name, config.max_optimizer_rmse, &mut failures);
    }
    if fixture.optimizer.name != "AdamW" {
        failures.push(format!(
            "upstream training fixture optimizer is {}, expected AdamW",
            fixture.optimizer.name
        ));
    }

    Ok(Npa2dTrainingParityReport {
        present: true,
        required: config.required,
        passed: failures.is_empty(),
        architecture: Some(fixture.architecture),
        batch_size: Some(fixture.batch_size),
        particle_count: Some(fixture.particle_count),
        comparisons,
        loss_differences,
        failures,
    })
}

fn validate_fixture_shape(fixture: &TrainingStepFixture) -> Result<(), std::io::Error> {
    let config = NpaConfig::growing_2d();
    let expected = [
        ("batch_size", fixture.batch_size, 1),
        ("spatial_dims", fixture.spatial_dims, config.spatial_dims),
        ("state_dims", fixture.state_dims, config.state_dims),
        ("hidden_dims", fixture.hidden_dims, config.hidden_dims),
        (
            "perception_dims",
            fixture.perception_dims,
            config.perception_dims(),
        ),
        ("update_dims", fixture.update_dims, config.update_dims()),
        (
            "positions",
            fixture.positions.len(),
            fixture.particle_count * 2,
        ),
        (
            "states",
            fixture.states.len(),
            fixture.particle_count * config.state_dims,
        ),
        (
            "update_mask",
            fixture.update_mask.len(),
            fixture.particle_count,
        ),
    ];
    for (name, actual, expected) in expected {
        if actual != expected {
            return Err(std::io::Error::other(format!(
                "upstream training fixture {name} {actual} != {expected}"
            )));
        }
    }
    Ok(())
}

fn flatten_spatial(values: &[[f32; 4]], dims: usize) -> Vec<f32> {
    values
        .iter()
        .flat_map(|value| value.iter().take(dims).copied())
        .collect()
}

fn insert_gradient_comparisons(
    out: &mut BTreeMap<String, TensorComparison>,
    prefix: &str,
    actual: &SupervisedGradients,
    expected: &ParameterFixture,
) -> Result<(), std::io::Error> {
    insert_comparison(out, &format!("{prefix}.w1"), &actual.w1, &expected.w1)?;
    insert_comparison(out, &format!("{prefix}.b1"), &actual.b1, &expected.b1)?;
    insert_comparison(out, &format!("{prefix}.w2"), &actual.w2, &expected.w2)?;
    Ok(())
}

fn insert_parameter_comparisons(
    out: &mut BTreeMap<String, TensorComparison>,
    prefix: &str,
    actual: &NpaWeights,
    expected: &ParameterFixture,
) -> Result<(), std::io::Error> {
    insert_comparison(out, &format!("{prefix}.w1"), &actual.w1, &expected.w1)?;
    insert_comparison(out, &format!("{prefix}.b1"), &actual.b1, &expected.b1)?;
    insert_comparison(out, &format!("{prefix}.w2"), &actual.w2, &expected.w2)?;
    Ok(())
}

fn insert_comparison(
    out: &mut BTreeMap<String, TensorComparison>,
    name: &str,
    actual: &[f32],
    expected: &[f32],
) -> Result<(), std::io::Error> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err(std::io::Error::other(format!(
            "training parity {name} shape mismatch: {} vs {}",
            actual.len(),
            expected.len()
        )));
    }
    let mut max_abs = 0.0_f64;
    let mut squared = 0.0_f64;
    let mut actual_squared = 0.0_f64;
    let mut expected_squared = 0.0_f64;
    let mut dot = 0.0_f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        let difference = actual - expected;
        max_abs = max_abs.max(difference.abs());
        squared += difference * difference;
        actual_squared += actual * actual;
        expected_squared += expected * expected;
        dot += actual * expected;
    }
    let denom = actual.len() as f64;
    out.insert(
        name.to_string(),
        TensorComparison {
            elements: actual.len(),
            max_abs,
            rmse: (squared / denom).sqrt(),
            relative_l2: squared.sqrt() / expected_squared.sqrt().max(f64::MIN_POSITIVE),
            cosine: dot / (actual_squared.sqrt() * expected_squared.sqrt()).max(f64::MIN_POSITIVE),
        },
    );
    Ok(())
}

fn insert_loss_difference(out: &mut BTreeMap<String, f64>, name: &str, actual: f32, expected: f32) {
    out.insert(name.to_string(), f64::from((actual - expected).abs()));
}

fn require_rmse(
    comparisons: &BTreeMap<String, TensorComparison>,
    name: &str,
    threshold: f64,
    failures: &mut Vec<String>,
) {
    let comparison = &comparisons[name];
    if !comparison.rmse.is_finite() || comparison.rmse > threshold {
        failures.push(format!(
            "training parity {name} RMSE {:.6e} exceeds {threshold:.6e}",
            comparison.rmse
        ));
    }
}

const fn default_feature_rmse() -> f64 {
    2.0e-4
}

const fn default_forward_rmse() -> f64 {
    2.0e-5
}

const fn default_loss_abs() -> f64 {
    2.0e-4
}

const fn default_gradient_cosine() -> f64 {
    0.999
}

const fn default_normalized_gradient_rmse() -> f64 {
    2.0e-4
}

const fn default_optimizer_rmse() -> f64 {
    2.0e-6
}
