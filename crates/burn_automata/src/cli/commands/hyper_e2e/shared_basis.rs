use crate::{LowRankAdapterGradients, supervised_backward};

use crate::cli::prelude::*;

use super::super::hyper_support::Hyper2dLoadedExample;
#[cfg(test)]
use super::super::hyper_support::Hyper2dSourceDescriptor;

#[derive(Clone, Copy)]
pub(super) struct SharedBasisFitConfig {
    pub(super) steps: usize,
    pub(super) report_interval: usize,
    pub(super) example_batch_size: usize,
    pub(super) adapter_l2_weight: f32,
    pub(super) seed: u64,
    pub(super) base_sgd: SgdConfig,
    pub(super) adapter_sgd: SgdConfig,
}

#[derive(Clone, Copy)]
struct SharedBasisStepStats {
    base_grad_norm: f32,
    base_grad_scale: f32,
    mean_adapter_grad_norm: f32,
    max_adapter_grad_norm: f32,
    examples_seen: usize,
}

pub(super) fn fit_shared_basis_and_adapters(
    base: &mut NpaModel,
    examples: &mut [HyperAdapterExample2d],
    loaded: &[Hyper2dLoadedExample],
    config: SharedBasisFitConfig,
) -> Result<CliHyper2dE2eSharedBasisFitReport, Box<dyn std::error::Error>> {
    validate_basis_examples(examples, loaded)?;
    validate_shared_basis_config(config)?;
    let rows = shared_basis_rows(base, loaded);
    let report_interval = config.report_interval.max(1);
    let example_batch_size =
        normalized_example_batch_size(config.example_batch_size, examples.len());
    let initial_loss = shared_basis_loss(base, examples, loaded, config.adapter_l2_weight)?;
    if config.steps == 0 {
        return Ok(CliHyper2dE2eSharedBasisFitReport {
            enabled: false,
            steps: 0,
            report_interval,
            rows,
            example_batch_size,
            adapter_l2_weight: config.adapter_l2_weight,
            seed: config.seed,
            base_sgd: config.base_sgd,
            adapter_sgd: config.adapter_sgd,
            initial_loss,
            final_loss: initial_loss,
            best_loss: initial_loss,
            best_step: 0,
            history: Vec::new(),
        });
    }

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut final_loss = initial_loss;
    let mut best_loss = initial_loss;
    let mut best_step = 0usize;
    let mut best_base = base.clone();
    let mut best_examples = examples.to_vec();
    let mut history = Vec::new();
    for step in 1..=config.steps {
        let indices = sample_example_indices(examples.len(), example_batch_size, &mut rng);
        let step_stats = shared_basis_train_step(
            base,
            examples,
            loaded,
            &indices,
            config.base_sgd,
            config.adapter_sgd,
            config.adapter_l2_weight,
        )?;
        if step == config.steps || step.is_multiple_of(report_interval) {
            final_loss = shared_basis_loss(base, examples, loaded, config.adapter_l2_weight)?;
            if final_loss < best_loss {
                best_loss = final_loss;
                best_step = step;
                best_base = base.clone();
                best_examples = examples.to_vec();
            }
            history.push(CliHyper2dE2eSharedBasisHistoryEntry {
                step,
                loss: final_loss,
                base_grad_norm: step_stats.base_grad_norm,
                base_grad_scale: step_stats.base_grad_scale,
                mean_adapter_grad_norm: step_stats.mean_adapter_grad_norm,
                max_adapter_grad_norm: step_stats.max_adapter_grad_norm,
                examples_seen: step_stats.examples_seen,
            });
        }
    }
    if best_loss < final_loss {
        *base = best_base;
        examples.clone_from_slice(&best_examples);
        final_loss = best_loss;
    }

    Ok(CliHyper2dE2eSharedBasisFitReport {
        enabled: true,
        steps: config.steps,
        report_interval,
        rows,
        example_batch_size,
        adapter_l2_weight: config.adapter_l2_weight,
        seed: config.seed,
        base_sgd: config.base_sgd,
        adapter_sgd: config.adapter_sgd,
        initial_loss,
        final_loss,
        best_loss,
        best_step,
        history,
    })
}

fn shared_basis_train_step(
    base: &mut NpaModel,
    examples: &mut [HyperAdapterExample2d],
    loaded: &[Hyper2dLoadedExample],
    indices: &[usize],
    base_sgd: SgdConfig,
    adapter_sgd: SgdConfig,
    adapter_l2_weight: f32,
) -> Result<SharedBasisStepStats, Box<dyn std::error::Error>> {
    validate_basis_examples(examples, loaded)?;
    validate_step_indices(indices, examples.len())?;
    let mut base_grads = zero_model_gradients(base);
    let example_scale = 1.0 / indices.len() as f32;
    let mut adapter_grad_sum = 0.0_f32;
    let mut adapter_grad_max = 0.0_f32;

    for &idx in indices {
        let loaded_example = &loaded[idx];
        let example = &mut examples[idx];
        let adapted = example.target_adapter.apply_to_model(base)?;
        let (full_grads, _) = supervised_backward(&adapted, &loaded_example.batch)?;
        let mut adapter_grads =
            project_low_rank_adapter_gradients(base, &example.target_adapter, &full_grads)?;
        add_adapter_l2_gradients(
            &example.target_adapter,
            &mut adapter_grads,
            adapter_l2_weight,
        );
        let adapter_step =
            apply_sgd_adapter_gradients(&mut example.target_adapter, &adapter_grads, adapter_sgd)?;
        add_scaled_model_gradients(&mut base_grads, &full_grads, example_scale);
        adapter_grad_sum += adapter_step.grad_norm;
        adapter_grad_max = adapter_grad_max.max(adapter_step.grad_norm);
    }

    let base_step = apply_sgd_gradients(base, &base_grads, base_sgd)?;
    Ok(SharedBasisStepStats {
        base_grad_norm: base_step.grad_norm,
        base_grad_scale: base_step.grad_scale,
        mean_adapter_grad_norm: adapter_grad_sum / indices.len() as f32,
        max_adapter_grad_norm: adapter_grad_max,
        examples_seen: indices.len(),
    })
}

fn shared_basis_loss(
    base: &NpaModel,
    examples: &[HyperAdapterExample2d],
    loaded: &[Hyper2dLoadedExample],
    adapter_l2_weight: f32,
) -> Result<f32, Box<dyn std::error::Error>> {
    validate_basis_examples(examples, loaded)?;
    let mut loss = 0.0_f32;
    for (example, loaded) in examples.iter().zip(loaded) {
        loss += supervised_adapter_loss(base, &example.target_adapter, &loaded.batch)?
            + adapter_l2_weight * adapter_l2_loss(&example.target_adapter);
    }
    Ok(loss / examples.len() as f32)
}

fn shared_basis_rows(base: &NpaModel, loaded: &[Hyper2dLoadedExample]) -> usize {
    let input_dims = base.config.perception_dims().max(1);
    loaded
        .iter()
        .map(|example| example.batch.features.len() / input_dims)
        .sum()
}

fn validate_basis_examples(
    examples: &[HyperAdapterExample2d],
    loaded: &[Hyper2dLoadedExample],
) -> Result<(), Box<dyn std::error::Error>> {
    if examples.is_empty() {
        return Err(std::io::Error::other("shared basis fitting requires examples").into());
    }
    if examples.len() != loaded.len() {
        return Err(
            std::io::Error::other("shared basis examples do not match loaded batches").into(),
        );
    }
    Ok(())
}

fn validate_shared_basis_config(
    config: SharedBasisFitConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    if !config.adapter_l2_weight.is_finite() || config.adapter_l2_weight < 0.0 {
        return Err(std::io::Error::other(
            "shared fit adapter L2 weight must be finite and non-negative",
        )
        .into());
    }
    Ok(())
}

fn validate_step_indices(
    indices: &[usize],
    examples_len: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if indices.is_empty() {
        return Err(std::io::Error::other("shared basis step requires examples").into());
    }
    if indices.iter().any(|idx| *idx >= examples_len) {
        return Err(std::io::Error::other("shared basis step index is out of range").into());
    }
    Ok(())
}

pub(super) fn normalized_example_batch_size(requested: usize, examples_len: usize) -> usize {
    if requested == 0 {
        examples_len
    } else {
        requested.min(examples_len).max(1)
    }
}

pub(super) fn sample_example_indices(
    examples_len: usize,
    example_batch_size: usize,
    rng: &mut StdRng,
) -> Vec<usize> {
    if example_batch_size.saturating_mul(4) < examples_len {
        let mut indices = std::collections::BTreeSet::new();
        while indices.len() < example_batch_size {
            indices.insert(rng.random_range(0..examples_len));
        }
        return indices.into_iter().collect();
    }
    let mut indices = (0..examples_len).collect::<Vec<_>>();
    indices.shuffle(rng);
    indices.truncate(example_batch_size);
    indices
}

pub(super) fn zero_model_gradients(model: &NpaModel) -> SupervisedGradients {
    SupervisedGradients {
        w1: vec![0.0; model.weights.w1.len()],
        b1: vec![0.0; model.weights.b1.len()],
        w2: vec![0.0; model.weights.w2.len()],
        b2: vec![0.0; model.weights.b2.len()],
        features: Vec::new(),
    }
}

pub(super) fn add_scaled_model_gradients(
    dst: &mut SupervisedGradients,
    src: &SupervisedGradients,
    scale: f32,
) {
    add_scaled_slice(&mut dst.w1, &src.w1, scale);
    add_scaled_slice(&mut dst.b1, &src.b1, scale);
    add_scaled_slice(&mut dst.w2, &src.w2, scale);
    add_scaled_slice(&mut dst.b2, &src.b2, scale);
}

fn add_scaled_slice(dst: &mut [f32], src: &[f32], scale: f32) {
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst += src * scale;
    }
}

pub(super) fn add_adapter_l2_gradients(
    adapter: &NpaLowRankAdapter,
    grads: &mut LowRankAdapterGradients,
    weight: f32,
) {
    if weight == 0.0 {
        return;
    }
    let scale = 2.0 * weight / adapter.parameter_count().max(1) as f32;
    add_l2_slice(&adapter.w1_down, &mut grads.w1_down, scale);
    add_l2_slice(&adapter.w1_up, &mut grads.w1_up, scale);
    add_l2_slice(&adapter.w2_down, &mut grads.w2_down, scale);
    add_l2_slice(&adapter.w2_up, &mut grads.w2_up, scale);
    add_l2_slice(&adapter.b1_delta, &mut grads.b1_delta, scale);
    add_l2_slice(&adapter.b2_delta, &mut grads.b2_delta, scale);
}

fn add_l2_slice(values: &[f32], grads: &mut [f32], scale: f32) {
    for (value, grad) in values.iter().zip(grads) {
        *grad += value * scale;
    }
}

pub(super) fn adapter_l2_loss(adapter: &NpaLowRankAdapter) -> f32 {
    let sum_sq = adapter
        .w1_down
        .iter()
        .chain(adapter.w1_up.iter())
        .chain(adapter.w2_down.iter())
        .chain(adapter.w2_up.iter())
        .chain(adapter.b1_delta.iter())
        .chain(adapter.b2_delta.iter())
        .map(|value| value * value)
        .sum::<f32>();
    sum_sq / adapter.parameter_count().max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loaded_example(
        slug: &str,
        base: &NpaModel,
        target: &NpaModel,
        seed: u64,
    ) -> Hyper2dLoadedExample {
        let condition = ConditionImage2d::from_rgb(1, 1, vec![1.0, 0.0, 0.0]).unwrap();
        let batch = feature_supervised_batch(
            base,
            SupervisedTarget::Teacher(target),
            FeatureBatchConfig {
                rows: 8,
                seed,
                amplitude: 0.25,
            },
        )
        .unwrap();
        Hyper2dLoadedExample {
            descriptor: Hyper2dSourceDescriptor {
                slug: slug.to_string(),
                title: None,
                group: None,
                condition_path: PathBuf::from(format!("{slug}.png")),
                target_path: PathBuf::from(format!("{slug}.bpk")),
                particles: None,
                seed_scale: None,
                update_prob: None,
            },
            condition,
            batch,
            rows: 8,
            particle_count: 8,
            rollout_steps: 1,
            rollouts: 1,
            update_prob: 1.0,
            seed_scale: 0.2,
            seed_mode: ParticleSeed::UniformCircle,
            seed,
        }
    }

    #[test]
    fn shared_basis_step_updates_base_and_adapter() {
        let config = NpaConfig::growing_2d();
        let mut base = NpaModel::upstream_seeded(config.clone(), 1);
        let target = NpaModel::upstream_seeded(config.clone(), 2);
        let mut examples = vec![HyperAdapterExample2d {
            condition: ConditionImage2d::from_rgb(1, 1, vec![1.0, 0.0, 0.0]).unwrap(),
            target_adapter: NpaLowRankAdapter::seeded(&config, 4, 4.0, 4),
        }];
        let loaded = vec![loaded_example("sample", &base, &target, 3)];
        let before_base_w1 = base.weights.w1.clone();
        let before_adapter = examples[0].target_adapter.to_parameter_vector();

        let report = shared_basis_train_step(
            &mut base,
            &mut examples,
            &loaded,
            &[0],
            SgdConfig {
                learning_rate: 1.0e-3,
                weight_decay: 0.0,
                grad_clip_norm: 1.0,
            },
            SgdConfig {
                learning_rate: 1.0e-3,
                weight_decay: 0.0,
                grad_clip_norm: 1.0,
            },
            0.0,
        )
        .unwrap();

        assert!(report.base_grad_norm.is_finite());
        assert!(report.mean_adapter_grad_norm.is_finite());
        assert_eq!(report.examples_seen, 1);
        assert_ne!(base.weights.w1, before_base_w1);
        assert_ne!(
            examples[0].target_adapter.to_parameter_vector(),
            before_adapter
        );
    }

    #[test]
    fn shared_basis_fit_reports_stochastic_batch_size() {
        let config = NpaConfig::growing_2d();
        let mut base = NpaModel::upstream_seeded(config.clone(), 1);
        let targets = [
            NpaModel::upstream_seeded(config.clone(), 2),
            NpaModel::upstream_seeded(config.clone(), 3),
        ];
        let mut examples = vec![
            HyperAdapterExample2d {
                condition: ConditionImage2d::from_rgb(1, 1, vec![1.0, 0.0, 0.0]).unwrap(),
                target_adapter: NpaLowRankAdapter::seeded(&config, 4, 4.0, 4),
            },
            HyperAdapterExample2d {
                condition: ConditionImage2d::from_rgb(1, 1, vec![0.0, 1.0, 0.0]).unwrap(),
                target_adapter: NpaLowRankAdapter::seeded(&config, 4, 4.0, 5),
            },
        ];
        let loaded = vec![
            loaded_example("first", &base, &targets[0], 7),
            loaded_example("second", &base, &targets[1], 8),
        ];

        let report = fit_shared_basis_and_adapters(
            &mut base,
            &mut examples,
            &loaded,
            SharedBasisFitConfig {
                steps: 2,
                report_interval: 1,
                example_batch_size: 1,
                adapter_l2_weight: 1.0e-4,
                seed: 9,
                base_sgd: SgdConfig {
                    learning_rate: 1.0e-3,
                    weight_decay: 0.0,
                    grad_clip_norm: 1.0,
                },
                adapter_sgd: SgdConfig {
                    learning_rate: 1.0e-3,
                    weight_decay: 0.0,
                    grad_clip_norm: 1.0,
                },
            },
        )
        .unwrap();

        assert!(report.enabled);
        assert_eq!(report.example_batch_size, 1);
        assert_eq!(report.adapter_l2_weight, 1.0e-4);
        assert_eq!(report.seed, 9);
        assert!(report.history.iter().all(|entry| entry.examples_seen == 1));
    }
}
