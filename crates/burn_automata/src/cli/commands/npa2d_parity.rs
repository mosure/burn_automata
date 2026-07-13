use std::process::Command as ProcessCommand;

use sha2::{Digest, Sha256};

use crate::cli::prelude::*;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Npa2dParityConfig {
    upstream: Npa2dParityUpstreamConfig,
    model: Npa2dParityModelConfig,
    validation: Npa2dParityValidationConfig,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Npa2dParityUpstreamConfig {
    repo_url: String,
    commit: String,
    cache_dir: PathBuf,
    config: PathBuf,
    checkpoint: PathBuf,
    fixture: PathBuf,
    #[serde(default)]
    require_fixture: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Npa2dParityModelConfig {
    bpk: PathBuf,
    target_image: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Npa2dParityValidationConfig {
    particles: usize,
    steps: usize,
    update_prob: f32,
    seed: u64,
    seed_scale: f32,
    seed_mode: String,
    target_points: usize,
    #[serde(default)]
    target_image_size: Option<usize>,
    target_threshold: f32,
    loss_image_size: usize,
    splat_sigma: f32,
    splat_loss_weight: f32,
    color_loss_weight: f32,
    density_loss_weight: f32,
    #[serde(default)]
    background_density_loss_weight: f32,
    #[serde(default)]
    foreground_density_loss_weight: f32,
    displacement_regularizer_weight: f32,
    overflow_regularizer_weight: f32,
    bound_regularizer_weight: f32,
    #[serde(default)]
    max_total_loss: Option<f32>,
    #[serde(default)]
    target_color_tolerance: Option<f32>,
}

#[derive(Serialize)]
struct Npa2dParityReport {
    passed: bool,
    failures: Vec<String>,
    config: String,
    upstream: Npa2dParityUpstreamReport,
    fixture: Npa2dParityFixtureReport,
    validation: Option<Npa2dParityValidationReport>,
}

#[derive(Serialize)]
struct Npa2dParityUpstreamReport {
    repo_url: String,
    expected_commit: String,
    cache_dir: String,
    cache_exists: bool,
    actual_commit: Option<String>,
    commit_matches: Option<bool>,
    config: String,
    config_exists: bool,
    checkpoint: String,
    checkpoint_exists: bool,
}

#[derive(Serialize)]
struct Npa2dParityFixtureReport {
    path: String,
    exists: bool,
    required: bool,
    target_point_count: Option<usize>,
    positions_sha256: Option<String>,
    colors_sha256: Option<String>,
    position_color_sorted_sha256: Option<String>,
    #[serde(skip_serializing)]
    pixel_size: Option<f32>,
    #[serde(skip_serializing)]
    threshold: Option<f32>,
    #[serde(skip_serializing)]
    position_color_rows: Option<Vec<[f32; 5]>>,
}

#[derive(Serialize)]
struct Npa2dParityValidationReport {
    model: Npa2dParityModelReport,
    target: Npa2dParityTargetReport,
    rollout: Npa2dParityRolloutReport,
    loss_config: Target2dLossConfig,
    loss: Target2dLossReport,
    max_total_loss: Option<f32>,
}

#[derive(Serialize)]
struct Npa2dParityModelReport {
    bpk: String,
    source: Option<String>,
    config: NpaConfig,
    hashgrid: burn_automata_kernels::HashGridConfig,
}

#[derive(Serialize)]
struct Npa2dParityTargetReport {
    loss_target_source: &'static str,
    image: String,
    source_width: usize,
    source_height: usize,
    target_points: usize,
    rust_image_target_points: usize,
    positions_sha256: String,
    colors_sha256: String,
    position_color_sorted_sha256: String,
    fixture_point_count_matches: Option<bool>,
    fixture_positions_match: Option<bool>,
    fixture_colors_match: Option<bool>,
    fixture_position_color_sorted_match: Option<bool>,
    fixture_position_set_matches: Option<bool>,
    fixture_color_rmse: Option<f32>,
    fixture_color_max_abs: Option<f32>,
    fixture_color_tolerance: Option<f32>,
}

#[derive(Serialize)]
struct Npa2dParityRolloutReport {
    particles: usize,
    steps: usize,
    update_prob: f32,
    seed: u64,
    seed_scale: f32,
    seed_mode: ParticleSeed,
}

pub(crate) fn run_validate_npa_2d_parity(
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    let Command::ValidateNpa2dParity { config, output } = command else {
        unreachable!("run_validate_npa_2d_parity called with the wrong command variant");
    };

    let text = std::fs::read_to_string(&config)?;
    let cfg: Npa2dParityConfig = toml::from_str(&text).map_err(|err| {
        std::io::Error::other(format!("failed to parse {}: {err}", config.display()))
    })?;

    let mut failures = Vec::new();
    let upstream = upstream_report(&cfg.upstream)?;
    if upstream.commit_matches == Some(false) {
        failures.push(format!(
            "upstream cache commit mismatch: expected {}, got {}",
            cfg.upstream.commit,
            upstream.actual_commit.as_deref().unwrap_or("unknown")
        ));
    }
    if upstream.cache_exists && !upstream.config_exists {
        failures.push(format!("upstream config is missing: {}", upstream.config));
    }
    if upstream.cache_exists && !upstream.checkpoint_exists {
        failures.push(format!(
            "upstream checkpoint is missing: {}",
            upstream.checkpoint
        ));
    }

    let fixture = fixture_report(&cfg.upstream.fixture, cfg.upstream.require_fixture)?;
    if cfg.upstream.require_fixture && !fixture.exists {
        failures.push(format!(
            "required upstream fixture is missing: {}; run scripts/fetch_selforg_npa.sh then scripts/export_selforg_npa_fixture.py",
            cfg.upstream.fixture.display()
        ));
        let report = Npa2dParityReport {
            passed: false,
            failures,
            config: config.display().to_string(),
            upstream,
            fixture,
            validation: None,
        };
        write_json_report(&output, &report)?;
        return Err(std::io::Error::other(format!(
            "NPA2D parity failed; wrote {}",
            output.display()
        ))
        .into());
    }

    let validation = validate_model_and_target(&cfg, &fixture)?;
    if let Some(max_total_loss) = cfg.validation.max_total_loss
        && validation.loss.total_loss > max_total_loss
    {
        failures.push(format!(
            "target2d loss {:.6} exceeds configured max_total_loss {:.6}",
            validation.loss.total_loss, max_total_loss
        ));
    }
    if validation.target.fixture_point_count_matches == Some(false) {
        failures.push("Rust target point count differs from upstream fixture".to_string());
    }
    if validation.target.fixture_position_set_matches == Some(false) {
        failures.push("Rust target positions differ from upstream fixture".to_string());
    }
    if let (Some(rmse), Some(tolerance)) = (
        validation.target.fixture_color_rmse,
        validation.target.fixture_color_tolerance,
    ) && rmse > tolerance
    {
        failures.push(format!(
            "Rust target colors differ from upstream fixture: rmse={rmse:.6} tolerance={tolerance:.6}"
        ));
    }

    let passed = failures.is_empty();
    let total_loss = validation.loss.total_loss;
    let target_points = validation.target.target_points;
    let report = Npa2dParityReport {
        passed,
        failures,
        config: config.display().to_string(),
        upstream,
        fixture,
        validation: Some(validation),
    };
    write_json_report(&output, &report)?;
    if !passed {
        return Err(std::io::Error::other(format!(
            "NPA2D parity failed; wrote {}",
            output.display()
        ))
        .into());
    }
    println!(
        "wrote {} target_points={} total_loss={:.6}",
        output.display(),
        target_points,
        total_loss
    );
    Ok(())
}

fn upstream_report(
    cfg: &Npa2dParityUpstreamConfig,
) -> Result<Npa2dParityUpstreamReport, Box<dyn std::error::Error>> {
    let cache_exists = cfg.cache_dir.join(".git").exists();
    let actual_commit = cache_exists
        .then(|| upstream_head(&cfg.cache_dir))
        .transpose()?;
    let commit_matches = actual_commit.as_ref().map(|commit| commit == &cfg.commit);
    let config_path = cfg.cache_dir.join(&cfg.config);
    let checkpoint_path = cfg.cache_dir.join(&cfg.checkpoint);
    Ok(Npa2dParityUpstreamReport {
        repo_url: cfg.repo_url.clone(),
        expected_commit: cfg.commit.clone(),
        cache_dir: cfg.cache_dir.display().to_string(),
        cache_exists,
        actual_commit,
        commit_matches,
        config: config_path.display().to_string(),
        config_exists: config_path.exists(),
        checkpoint: checkpoint_path.display().to_string(),
        checkpoint_exists: checkpoint_path.exists(),
    })
}

fn upstream_head(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let output = ProcessCommand::new("git")
        .args(["-C", &path.display().to_string(), "rev-parse", "HEAD"])
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "failed to read upstream git head in {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn fixture_report(
    path: &Path,
    required: bool,
) -> Result<Npa2dParityFixtureReport, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(Npa2dParityFixtureReport {
            path: path.display().to_string(),
            exists: false,
            required,
            target_point_count: None,
            positions_sha256: None,
            colors_sha256: None,
            position_color_sorted_sha256: None,
            pixel_size: None,
            threshold: None,
            position_color_rows: None,
        });
    }
    let value: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let target = value.get("target").ok_or_else(|| {
        std::io::Error::other(format!(
            "upstream fixture {} does not contain target",
            path.display()
        ))
    })?;
    let position_color_rows = fixture_position_color_rows(target);
    let position_color_sorted_sha256 = position_color_rows
        .clone()
        .map(|mut rows| hash_sorted_rows(&mut rows));
    Ok(Npa2dParityFixtureReport {
        path: path.display().to_string(),
        exists: true,
        required,
        target_point_count: target
            .get("point_count")
            .and_then(serde_json::Value::as_u64)
            .map(|value| value as usize),
        positions_sha256: target
            .get("positions_sha256")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
        colors_sha256: target
            .get("colors_sha256")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
        position_color_sorted_sha256,
        pixel_size: target
            .get("pixel_size")
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32),
        threshold: target
            .get("threshold")
            .and_then(serde_json::Value::as_f64)
            .map(|value| value as f32),
        position_color_rows,
    })
}

fn validate_model_and_target(
    cfg: &Npa2dParityConfig,
    fixture: &Npa2dParityFixtureReport,
) -> Result<Npa2dParityValidationReport, Box<dyn std::error::Error>> {
    let seed_mode = SeedModeArg::from_str(&cfg.validation.seed_mode, true)
        .map_err(|err| {
            std::io::Error::other(format!(
                "invalid validation.seed_mode {:?}: {err}",
                cfg.validation.seed_mode
            ))
        })?
        .into();
    let loss_config = super::target2d::target2d_loss_config(
        cfg.validation.loss_image_size,
        cfg.validation.splat_sigma,
        true,
        cfg.validation.splat_loss_weight,
        cfg.validation.color_loss_weight,
        cfg.validation.density_loss_weight,
        cfg.validation.background_density_loss_weight,
        cfg.validation.foreground_density_loss_weight,
        cfg.validation.displacement_regularizer_weight,
        cfg.validation.overflow_regularizer_weight,
        cfg.validation.bound_regularizer_weight,
    )?;
    let rust_image_target = super::target2d::load_target_image_2d_adaptive(
        &cfg.model.target_image,
        cfg.validation.target_threshold,
        cfg.validation.target_points,
        cfg.validation.target_image_size,
    )?;
    let fixture_comparison = fixture.position_color_rows.as_ref().map(|fixture_rows| {
        compare_position_color_rows(fixture_rows, &position_color_rows(&rust_image_target))
    });
    let (target, loss_target_source) = fixture_target(fixture, &rust_image_target)
        .map(|target| (target, "upstream_fixture"))
        .unwrap_or_else(|| (rust_image_target.clone(), "rust_image"));
    let positions_sha256 = hash_f32_rows(&target.positions);
    let colors_sha256 = hash_f32_rows(&target.colors);
    let position_color_rows = position_color_rows(&target);
    let position_color_sorted_sha256 = {
        let mut rows = position_color_rows.clone();
        hash_sorted_rows(&mut rows)
    };
    let color_tolerance = fixture_comparison
        .as_ref()
        .map(|_| cfg.validation.target_color_tolerance.unwrap_or(0.02));
    let target_report = Npa2dParityTargetReport {
        loss_target_source,
        image: cfg.model.target_image.display().to_string(),
        source_width: target.source_width,
        source_height: target.source_height,
        target_points: target.point_count(),
        rust_image_target_points: rust_image_target.point_count(),
        fixture_point_count_matches: fixture
            .target_point_count
            .map(|count| count == rust_image_target.point_count()),
        fixture_positions_match: fixture
            .positions_sha256
            .as_ref()
            .map(|hash| hash == &positions_sha256),
        fixture_colors_match: fixture
            .colors_sha256
            .as_ref()
            .map(|hash| hash == &colors_sha256),
        fixture_position_color_sorted_match: fixture
            .position_color_sorted_sha256
            .as_ref()
            .map(|hash| hash == &position_color_sorted_sha256),
        fixture_position_set_matches: fixture_comparison
            .as_ref()
            .map(|comparison| comparison.position_max_abs <= 1.0e-6),
        fixture_color_rmse: fixture_comparison
            .as_ref()
            .map(|comparison| comparison.color_rmse),
        fixture_color_max_abs: fixture_comparison
            .as_ref()
            .map(|comparison| comparison.color_max_abs),
        fixture_color_tolerance: color_tolerance,
        positions_sha256,
        colors_sha256,
        position_color_sorted_sha256,
    };

    let manifest = crate::import::load_manifest(&cfg.model.bpk)?;
    if manifest.config.spatial_dims != 2 || manifest.hashgrid.dim != 2 {
        return Err(std::io::Error::other("NPA2D parity requires a 2D BPK model").into());
    }
    let model_report = Npa2dParityModelReport {
        bpk: cfg.model.bpk.display().to_string(),
        source: manifest.source.clone(),
        config: manifest.config.clone(),
        hashgrid: manifest.hashgrid.clone(),
    };
    let hashgrid = manifest.hashgrid.clone();
    let model = manifest.into_model();
    let loss = super::target2d::evaluate_target2d_loaded_model_loss(
        &model,
        &hashgrid,
        &target,
        loss_config,
        cfg.validation.particles,
        cfg.validation.steps,
        cfg.validation.update_prob,
        cfg.validation.seed,
        cfg.validation.seed_scale,
        seed_mode,
    )?;

    Ok(Npa2dParityValidationReport {
        model: model_report,
        target: target_report,
        rollout: Npa2dParityRolloutReport {
            particles: cfg.validation.particles,
            steps: cfg.validation.steps,
            update_prob: cfg.validation.update_prob,
            seed: cfg.validation.seed,
            seed_scale: cfg.validation.seed_scale,
            seed_mode,
        },
        loss_config,
        loss,
        max_total_loss: cfg.validation.max_total_loss,
    })
}

fn fixture_target(
    fixture: &Npa2dParityFixtureReport,
    local_target: &TargetImage2d,
) -> Option<TargetImage2d> {
    let rows = fixture.position_color_rows.as_ref()?;
    let mut positions = Vec::with_capacity(rows.len());
    let mut colors = Vec::with_capacity(rows.len());
    for row in rows {
        positions.push([row[0], row[1]]);
        colors.push([row[2], row[3], row[4]]);
    }
    Some(TargetImage2d {
        source_width: local_target.source_width,
        source_height: local_target.source_height,
        positions,
        colors,
        pixel_size: fixture.pixel_size.unwrap_or(local_target.pixel_size),
        threshold: fixture.threshold.unwrap_or(local_target.threshold),
        aabb: local_target.aabb,
    })
}

#[derive(Clone, Copy)]
struct FixtureTargetComparison {
    position_max_abs: f32,
    color_rmse: f32,
    color_max_abs: f32,
}

fn fixture_position_color_rows(target: &serde_json::Value) -> Option<Vec<[f32; 5]>> {
    let positions = target.get("positions")?.as_array()?;
    let colors = target.get("colors")?.as_array()?;
    if positions.len() != colors.len() {
        return None;
    }
    let mut rows = Vec::with_capacity(positions.len());
    for (position, color) in positions.iter().zip(colors) {
        let position = position.as_array()?;
        let color = color.as_array()?;
        if position.len() != 2 || color.len() != 3 {
            return None;
        }
        rows.push([
            position[0].as_f64()? as f32,
            position[1].as_f64()? as f32,
            color[0].as_f64()? as f32,
            color[1].as_f64()? as f32,
            color[2].as_f64()? as f32,
        ]);
    }
    Some(rows)
}

fn position_color_rows(target: &TargetImage2d) -> Vec<[f32; 5]> {
    target
        .positions
        .iter()
        .zip(&target.colors)
        .map(|(position, color)| [position[0], position[1], color[0], color[1], color[2]])
        .collect::<Vec<_>>()
}

fn compare_position_color_rows(
    fixture_rows: &[[f32; 5]],
    target_rows: &[[f32; 5]],
) -> FixtureTargetComparison {
    if fixture_rows.len() != target_rows.len() {
        return FixtureTargetComparison {
            position_max_abs: f32::INFINITY,
            color_rmse: f32::INFINITY,
            color_max_abs: f32::INFINITY,
        };
    }
    let mut fixture_rows = fixture_rows.to_vec();
    let mut target_rows = target_rows.to_vec();
    sort_position_color_rows(&mut fixture_rows);
    sort_position_color_rows(&mut target_rows);

    let mut position_max_abs = 0.0_f32;
    let mut color_sq_sum = 0.0_f32;
    let mut color_count = 0usize;
    let mut color_max_abs = 0.0_f32;
    for (fixture, target) in fixture_rows.iter().zip(&target_rows) {
        for channel in 0..2 {
            position_max_abs = position_max_abs.max((fixture[channel] - target[channel]).abs());
        }
        for channel in 2..5 {
            let diff = fixture[channel] - target[channel];
            color_sq_sum += diff * diff;
            color_count += 1;
            color_max_abs = color_max_abs.max(diff.abs());
        }
    }
    FixtureTargetComparison {
        position_max_abs,
        color_rmse: (color_sq_sum / color_count.max(1) as f32).sqrt(),
        color_max_abs,
    }
}

fn hash_sorted_rows<const N: usize>(rows: &mut [[f32; N]]) -> String {
    rows.sort_by(|left, right| {
        left.iter()
            .zip(right)
            .map(|(left, right)| left.total_cmp(right))
            .find(|ordering| !ordering.is_eq())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hash_f32_rows(rows)
}

fn sort_position_color_rows(rows: &mut [[f32; 5]]) {
    rows.sort_by(|left, right| {
        left[0]
            .total_cmp(&right[0])
            .then(left[1].total_cmp(&right[1]))
            .then(left[2].total_cmp(&right[2]))
            .then(left[3].total_cmp(&right[3]))
            .then(left[4].total_cmp(&right[4]))
    });
}

fn hash_f32_rows<const N: usize>(rows: &[[f32; N]]) -> String {
    let mut hasher = Sha256::new();
    for row in rows {
        for value in row {
            hasher.update(value.to_ne_bytes());
        }
    }
    hex_digest(hasher.finalize().as_slice())
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn write_json_report<T: Serialize>(
    path: &Path,
    report: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verified_npa2d_parity_configs_parse() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        for name in ["lizard_smoke.toml", "lizard_full.toml"] {
            let path = repo_root.join("configs/verified/2d/parity").join(name);
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
            toml::from_str::<Npa2dParityConfig>(&text)
                .unwrap_or_else(|err| panic!("failed to parse {}: {err}", path.display()));
        }
    }
}
