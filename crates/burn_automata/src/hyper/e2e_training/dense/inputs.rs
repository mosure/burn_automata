//! Target preparation, condition loading, sampling, and device particle pools.

use super::*;

    pub(super) const E2E_CONDITION_DIAGNOSTIC_ROWS: usize = 32;

    pub(super) fn condition_diagnostic_indices(examples: usize) -> Vec<usize> {
        let rows = examples.min(E2E_CONDITION_DIAGNOSTIC_ROWS);
        match rows {
            0 => Vec::new(),
            1 => vec![0],
            _ => (0..rows)
                .map(|row| row.saturating_mul(examples - 1) / (rows - 1))
                .collect(),
        }
    }

    pub(super) fn burn_targets(
        examples: &[DirectBasisExample],
        config: DirectBasisTrainConfig,
        device: &BurnDevice,
    ) -> AutomataResult<Vec<BurnTargetExample>> {
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let pixel_xy = tensor(
            pixel_xy_values(config.loss_config.image_size),
            [pixels, 2],
            device,
        );
        examples
            .iter()
            .map(|example| {
                let render = render_target_2d_splat(&example.target, config.loss_config)?;
                let foreground = target_2d_foreground_mask(&example.target, config.loss_config)?;
                let foreground_scale = pixels as f32 / foreground.iter().sum::<f32>().max(1.0);
                let target_mean = example.target.mean_position();
                let target_positions = example
                    .target
                    .positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]])
                    .collect::<Vec<_>>();
                Ok(BurnTargetExample {
                    target_rgb: tensor(render.rgb, [pixels, 3], device),
                    target_density: tensor(render.density, [pixels, 1], device),
                    target_foreground: tensor(foreground, [pixels, 1], device),
                    target_foreground_scale: foreground_scale,
                    target_mean: tensor([target_mean[0], target_mean[1]].to_vec(), [1, 2], device),
                    target_positions: tensor(
                        target_positions,
                        [example.target.positions.len(), 2],
                        device,
                    ),
                    pixel_xy: pixel_xy.clone(),
                    pixel_size: example.target.pixel_size,
                    target_points: example.target.point_count(),
                    particle_count: example.particle_count.unwrap_or(config.rollout_particles),
                    update_prob: example.update_prob.unwrap_or(config.update_prob),
                    seed_scale: example.seed_scale.unwrap_or(config.seed_scale),
                    target_cpu: example.target.clone(),
                })
            })
            .collect()
    }

    pub(super) fn burn_e2e_targets_for_indices_with_runtime(
        examples: &[BurnE2eRolloutExample],
        indices: &[usize],
        config: BurnE2eRolloutTrainConfig,
        device: &BurnDevice,
        particle_count: Option<usize>,
        update_prob: Option<f32>,
    ) -> AutomataResult<Vec<BurnTargetExample>> {
        let direct_config = direct_config_view(config);
        let pixel_xy = burn_e2e_pixel_xy(config, device);
        indices
            .iter()
            .map(|idx| {
                examples.get(*idx).ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "HyperNPA e2e target index out of bounds".to_string(),
                    )
                })
            })
            .collect::<AutomataResult<Vec<_>>>()?
            .into_iter()
            .map(|example| {
                prepare_e2e_cpu_target(
                    BurnE2eCpuTargetInput {
                        target: example.target.clone(),
                        particle_count: particle_count.unwrap_or(example.particle_count).max(1),
                        update_prob: update_prob.unwrap_or(example.update_prob),
                        seed_scale: example.seed_scale,
                    },
                    direct_config,
                )
                .map(|prepared| prepared.into_burn(&pixel_xy, device))
            })
            .collect()
    }

    pub(super) fn burn_e2e_pixel_xy(config: BurnE2eRolloutTrainConfig, device: &BurnDevice) -> Tensor2 {
        let image_size = direct_config_view(config).loss_config.image_size;
        tensor(
            pixel_xy_values(image_size),
            [image_size * image_size, 2],
            device,
        )
    }

    pub(super) fn e2e_target_cache_bytes(
        examples: &[BurnE2eRolloutExample],
        config: BurnE2eRolloutTrainConfig,
    ) -> usize {
        let pixels = config
            .loss_config
            .image_size
            .saturating_mul(config.loss_config.image_size);
        examples.iter().fold(0usize, |bytes, example| {
            let floats = pixels
                .saturating_mul(5)
                .saturating_add(2)
                .saturating_add(example.target.point_count_hint().saturating_mul(2));
            bytes.saturating_add(floats.saturating_mul(std::mem::size_of::<f32>()))
        })
    }

    pub(super) fn split_e2e_condition_cache_budget(
        total_bytes: usize,
        train_examples: usize,
        holdout_examples: usize,
    ) -> (usize, usize) {
        let examples = train_examples.saturating_add(holdout_examples);
        if examples == 0 {
            return (0, 0);
        }
        let train_bytes = total_bytes
            .saturating_mul(train_examples)
            .checked_div(examples)
            .unwrap_or(0);
        (train_bytes, total_bytes.saturating_sub(train_bytes))
    }

    pub(super) fn burn_e2e_target_cache(
        examples: &[BurnE2eRolloutExample],
        config: BurnE2eRolloutTrainConfig,
        pixel_xy: &Tensor2,
        device: &BurnDevice,
    ) -> AutomataResult<Vec<BurnTargetExample>> {
        let direct_config = direct_config_view(config);
        let prepared = examples
            .par_iter()
            .map(|example| {
                prepare_e2e_cpu_target(
                    BurnE2eCpuTargetInput {
                        target: example.target.clone(),
                        particle_count: example.particle_count,
                        update_prob: example.update_prob,
                        seed_scale: example.seed_scale,
                    },
                    direct_config,
                )
            })
            .collect::<AutomataResult<Vec<_>>>()?;
        burn_e2e_prepared_targets_to_burn(prepared, pixel_xy, device)
    }

    impl BurnE2eTargetDeviceCache {
        pub(super) fn new(max_bytes: usize, expected_entries: usize) -> Self {
            Self {
                entries: HashMap::with_capacity(expected_entries),
                max_bytes,
                resident_bytes: 0,
                access_clock: 0,
                hits: 0,
                misses: 0,
                inserts: 0,
                evictions: 0,
            }
        }

        pub(super) fn cached_identities(&self) -> HashSet<usize> {
            self.entries.keys().copied().collect()
        }

        pub(super) fn insert_prepared(
            &mut self,
            identities: &[usize],
            targets: Vec<BurnE2ePreparedTargetExample>,
            protected_identities: &[usize],
            pixel_xy: &Tensor2,
            device: &BurnDevice,
        ) -> AutomataResult<()> {
            if identities.len() != targets.len() {
                return Err(AutomataError::InvalidArgument(format!(
                    "HyperNPA target cache insert identity/target mismatch: {} identities for {} targets",
                    identities.len(),
                    targets.len(),
                )));
            }
            let protected = protected_identities
                .iter()
                .copied()
                .collect::<HashSet<_>>();
            for (identity, target) in identities.iter().copied().zip(targets) {
                self.access_clock = self.access_clock.saturating_add(1);
                if let Some(entry) = self.entries.get_mut(&identity) {
                    entry.last_access = self.access_clock;
                    continue;
                }
                let bytes = target.device_bytes();
                if bytes > self.max_bytes {
                    return Err(AutomataError::InvalidArgument(format!(
                        "HyperNPA target identity {identity} requires {bytes} device bytes, exceeding the bounded target cache limit {}",
                        self.max_bytes,
                    )));
                }
                while self.resident_bytes.saturating_add(bytes) > self.max_bytes {
                    let evict_identity = self
                        .entries
                        .iter()
                        .filter(|(candidate, _)| !protected.contains(candidate))
                        .min_by_key(|(_, entry)| entry.last_access)
                        .map(|(candidate, _)| *candidate)
                        .ok_or_else(|| {
                            AutomataError::InvalidArgument(format!(
                                "HyperNPA target cache cannot fit the active request within {} bytes; increase target_device_cache_max_bytes",
                                self.max_bytes,
                            ))
                        })?;
                    let evicted = self
                        .entries
                        .remove(&evict_identity)
                        .expect("selected target cache eviction must exist");
                    self.resident_bytes = self.resident_bytes.saturating_sub(evicted.bytes);
                    self.evictions = self.evictions.saturating_add(1);
                }
                self.resident_bytes = self.resident_bytes.saturating_add(bytes);
                self.entries.insert(
                    identity,
                    BurnE2eTargetDeviceCacheEntry {
                        target: target.into_burn(pixel_xy, device),
                        bytes,
                        last_access: self.access_clock,
                    },
                );
                self.inserts = self.inserts.saturating_add(1);
            }
            Ok(())
        }

        pub(super) fn select(
            &mut self,
            indices: &[usize],
        ) -> AutomataResult<(Vec<BurnTargetExample>, Vec<usize>)> {
            let (unique, expansion) = deduplicate_condition_indices(indices);
            let mut targets = Vec::with_capacity(unique.len());
            for identity in unique {
                self.access_clock = self.access_clock.saturating_add(1);
                let Some(entry) = self.entries.get_mut(&identity) else {
                    self.misses = self.misses.saturating_add(1);
                    return Err(AutomataError::InvalidArgument(format!(
                        "HyperNPA target identity {identity} was not populated before device-cache selection"
                    )));
                };
                entry.last_access = self.access_clock;
                targets.push(entry.target.clone());
                self.hits = self.hits.saturating_add(1);
            }
            Ok((targets, expansion))
        }

        pub(super) fn metrics(&self) -> serde_json::Value {
            let lookups = self.hits.saturating_add(self.misses);
            json!({
                "mode": "bounded-device-lru",
                "max_bytes": self.max_bytes,
                "resident_bytes": self.resident_bytes,
                "resident_rows": self.entries.len(),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": if lookups == 0 {
                    0.0
                } else {
                    self.hits as f64 / lookups as f64
                },
                "inserts": self.inserts,
                "evictions": self.evictions,
            })
        }
    }

    pub(super) fn e2e_target_prefetch_mode(
        complete_cache: bool,
        bounded_cache: Option<&BurnE2eTargetDeviceCache>,
        skip_targets: bool,
    ) -> BurnE2eTargetPrefetchMode {
        if complete_cache || skip_targets {
            BurnE2eTargetPrefetchMode::Skip
        } else if let Some(cache) = bounded_cache {
            BurnE2eTargetPrefetchMode::Missing(cache.cached_identities())
        } else {
            BurnE2eTargetPrefetchMode::All
        }
    }

    pub(super) fn spawn_e2e_cpu_batch_prefetch(
        examples: &[BurnE2eRolloutExample],
        conditions: &BurnE2eConditionCache,
        indices: Vec<usize>,
        config: BurnE2eRolloutTrainConfig,
        target_mode: BurnE2eTargetPrefetchMode,
    ) -> AutomataResult<BurnE2eCpuBatchPrefetch> {
        let mut target_inputs = Vec::with_capacity(indices.len());
        let (unique_target_indices, target_expansion) = deduplicate_condition_indices(&indices);
        let target_identities = unique_target_indices
            .iter()
            .copied()
            .filter(|identity| match &target_mode {
                BurnE2eTargetPrefetchMode::All => true,
                BurnE2eTargetPrefetchMode::Missing(cached) => !cached.contains(identity),
                BurnE2eTargetPrefetchMode::Skip => false,
            })
            .collect::<Vec<_>>();
        target_inputs.reserve(target_identities.len());
        for &idx in &target_identities {
                let example = examples.get(idx).ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "HyperNPA e2e prefetch target index out of bounds".to_string(),
                    )
                })?;
                target_inputs.push(BurnE2eCpuTargetInput {
                    target: example.target.clone(),
                    particle_count: example.particle_count,
                    update_prob: example.update_prob,
                    seed_scale: example.seed_scale,
                });
        }
        let (unique_condition_indices, condition_expansion) =
            deduplicate_condition_indices(&indices);
        let condition_paths = if config.amortization_substrate_only {
            None
        } else {
            conditions
                .dynamic_dino_paths_for_indices(&unique_condition_indices)?
                .map(|(identities, paths)| (paths, identities, condition_expansion))
        };
        let pending_indices = indices.clone();
        Ok(BurnE2eCpuBatchPrefetch {
            indices: pending_indices,
            handle: thread::spawn(move || {
                prepare_e2e_cpu_batch(
                    indices,
                    target_identities,
                    target_inputs,
                    target_expansion,
                    condition_paths,
                    config,
                )
            }),
        })
    }

    pub(super) fn join_e2e_cpu_batch_prefetch(
        handle: BurnE2eCpuBatchPrefetch,
    ) -> AutomataResult<BurnE2ePreparedCpuBatch> {
        handle
            .handle
            .join()
            .map_err(|_| {
                AutomataError::InvalidArgument("HyperNPA e2e CPU prefetch panicked".to_string())
            })?
            .map_err(AutomataError::InvalidArgument)
    }

    pub(super) fn prepare_e2e_cpu_batch(
        indices: Vec<usize>,
        target_identities: Vec<usize>,
        target_inputs: Vec<BurnE2eCpuTargetInput>,
        target_expansion: Vec<usize>,
        condition_paths: Option<(Vec<PathBuf>, Vec<usize>, Vec<usize>)>,
        config: BurnE2eRolloutTrainConfig,
    ) -> Result<BurnE2ePreparedCpuBatch, String> {
        let direct_config = direct_config_view(config);
        let (targets, prepared_dino) = rayon::join(
            move || {
                target_inputs
                    .into_par_iter()
                    .map(|input| {
                        prepare_e2e_cpu_target(input, direct_config).map_err(|err| err.to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()
            },
            move || match condition_paths {
                Some((paths, identities, expansion)) => {
                    prepare_dino_condition_batch_for_prefetch(
                        paths,
                        identities,
                        expansion,
                        config.dino_image_size,
                    )
                    .map(Some)
                }
                None => Ok(None),
            },
        );
        let targets = targets?;
        let prepared_dino = prepared_dino?;
        Ok(BurnE2ePreparedCpuBatch {
            indices,
            target_identities,
            targets,
            target_expansion,
            prepared_dino,
        })
    }

    pub(super) fn deduplicate_condition_indices(indices: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let mut unique = Vec::new();
        let mut rows = HashMap::new();
        let mut expansion = Vec::with_capacity(indices.len());
        for &identity in indices {
            let row = *rows.entry(identity).or_insert_with(|| {
                let row = unique.len();
                unique.push(identity);
                row
            });
            expansion.push(row);
        }
        (unique, expansion)
    }

    #[cfg(feature = "dino")]
    pub(super) fn next_e2e_dino_cache_slot(
        slot_identities: &[Option<usize>],
        protected_identities: &HashSet<usize>,
        next_evict: &mut usize,
    ) -> Option<usize> {
        let capacity = slot_identities.len();
        if capacity == 0 {
            return None;
        }
        if let Some(slot) = slot_identities.iter().position(Option::is_none) {
            *next_evict = (slot + 1) % capacity;
            return Some(slot);
        }
        for _ in 0..capacity {
            let slot = *next_evict % capacity;
            *next_evict = (slot + 1) % capacity;
            let identity = slot_identities[slot]
                .expect("a full DINO cache slot must have an identity");
            if !protected_identities.contains(&identity) {
                return Some(slot);
            }
        }
        None
    }

    pub(super) fn prepare_e2e_cpu_target(
        input: BurnE2eCpuTargetInput,
        config: DirectBasisTrainConfig,
    ) -> AutomataResult<BurnE2ePreparedTargetExample> {
        let target = input.target.load()?;
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let render = render_target_2d_splat(&target, config.loss_config)?;
        let foreground = target_2d_foreground_mask(&target, config.loss_config)?;
        let target_foreground_scale = pixels as f32 / foreground.iter().sum::<f32>().max(1.0);
        let target_mean = target.mean_position();
        let target_positions = target
            .positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect::<Vec<_>>();
        Ok(BurnE2ePreparedTargetExample {
            target_rgb: render.rgb,
            target_density: render.density,
            target_foreground: foreground,
            target_foreground_scale,
            target_mean,
            target_positions,
            pixel_size: target.pixel_size,
            target_points: target.point_count(),
            particle_count: input.particle_count.max(1),
            update_prob: input.update_prob,
            seed_scale: input.seed_scale,
            target_cpu: target,
        })
    }

    pub(super) fn burn_e2e_prepared_targets_to_burn(
        targets: Vec<BurnE2ePreparedTargetExample>,
        pixel_xy: &Tensor2,
        device: &BurnDevice,
    ) -> AutomataResult<Vec<BurnTargetExample>> {
        targets
            .into_iter()
            .map(|target| Ok(target.into_burn(pixel_xy, device)))
            .collect()
    }

    impl BurnE2ePreparedTargetExample {
        pub(super) fn device_bytes(&self) -> usize {
            self.target_rgb
                .len()
                .saturating_add(self.target_density.len())
                .saturating_add(self.target_foreground.len())
                .saturating_add(self.target_mean.len())
                .saturating_add(self.target_positions.len())
                .saturating_mul(std::mem::size_of::<f32>())
        }

        pub(super) fn into_burn(self, pixel_xy: &Tensor2, device: &BurnDevice) -> BurnTargetExample {
            let pixels = self.target_rgb.len() / 3;
            let target_position_count = self.target_positions.len() / 2;
            BurnTargetExample {
                target_rgb: tensor(self.target_rgb, [pixels, 3], device),
                target_density: tensor(self.target_density, [pixels, 1], device),
                target_foreground: tensor(self.target_foreground, [pixels, 1], device),
                target_foreground_scale: self.target_foreground_scale,
                target_mean: tensor(self.target_mean.to_vec(), [1, 2], device),
                target_positions: tensor(
                    self.target_positions,
                    [target_position_count, 2],
                    device,
                ),
                pixel_xy: pixel_xy.clone(),
                pixel_size: self.pixel_size,
                target_points: self.target_points,
                particle_count: self.particle_count,
                update_prob: self.update_prob,
                seed_scale: self.seed_scale,
                target_cpu: self.target_cpu,
            }
        }
    }

    pub(super) fn e2e_cpu_prefetch_depth(batch_size: usize, steps: usize) -> usize {
        if steps <= 1 {
            return 1;
        }
        let depth = if batch_size >= 256 { 2 } else { 4 };
        depth.min(steps).max(1)
    }

    #[cfg(feature = "dino")]
    pub(super) fn prepare_dino_condition_batch_for_prefetch(
        paths: Vec<PathBuf>,
        encoded_identities: Vec<usize>,
        expansion: Vec<usize>,
        image_size: usize,
    ) -> Result<BurnE2ePreparedDinoBatch, String> {
        if paths.len() != encoded_identities.len() {
            return Err("DINO prefetch paths and identities have different lengths".to_string());
        }
        let encoded_rows = paths.len();
        let images = paths
            .into_par_iter()
            .map(|path| load_dino_condition_image(&path).map_err(|err| err.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let prepared = DinoVitsPreparedConditionBatch::from_conditions(&images, image_size)
            .map_err(|err| err.to_string())?;
        Ok(BurnE2ePreparedDinoBatch {
            prepared,
            encoded_rows,
            encoded_identities,
            expansion,
        })
    }

    #[cfg(not(feature = "dino"))]
    pub(super) fn prepare_dino_condition_batch_for_prefetch(
        _paths: Vec<PathBuf>,
        _encoded_identities: Vec<usize>,
        _expansion: Vec<usize>,
        _image_size: usize,
    ) -> Result<BurnE2ePreparedDinoBatch, String> {
        Err("DINO prefetch requires the dino feature".to_string())
    }

    pub(super) fn direct_config_view(config: BurnE2eRolloutTrainConfig) -> DirectBasisTrainConfig {
        DirectBasisTrainConfig {
            steps: config.steps,
            report_interval: config.report_interval,
            example_batch_size: config.example_batch_size,
            tbptt_chunk_steps: config.tbptt_chunk_steps,
            loss_on_final_chunk_only: config.loss_on_final_chunk_only,
            use_particle_pool: config.use_particle_pool,
            pool_size: config.pool_slots_per_example.max(1),
            inject_seed_interval: config.inject_seed_interval,
            brush_size: config.brush_size,
            stopgrad_pos: config.stopgrad_pos,
            stopgrad_state: config.stopgrad_state,
            rollout_particles: config.rollout_particles,
            rollout_step_min: config.rollout_step_min,
            rollout_steps: config.rollout_steps,
            update_prob: config.update_prob,
            seed: config.seed,
            seed_scale: config.seed_scale,
            seed_mode: config.seed_mode,
            grid_eps: config.grid_eps,
            motion_scale: config.motion_scale,
            loss_config: config.loss_config,
            target2d_loss_backend: config.target2d_loss_backend,
            perception_backend: config.perception_backend,
            per_parameter_grad_normalization: config.per_parameter_grad_normalization,
            base_sgd: SgdConfig {
                learning_rate: config.base_optimizer.learning_rate,
                weight_decay: config.base_optimizer.weight_decay,
                grad_clip_norm: config.base_optimizer.grad_clip_norm,
            },
            adapter_sgd: SgdConfig {
                learning_rate: config.generator_optimizer.learning_rate,
                weight_decay: config.generator_optimizer.weight_decay,
                grad_clip_norm: config.generator_optimizer.grad_clip_norm,
            },
            adapter_l2_weight: 0.0,
            update_base: config.shared_base_trainable,
            eval_examples: 0,
            eval_interval: 0,
            eval_batch_size: 1,
            eval_seed: config.seed,
            system_memory_budget_gb: config.system_memory_budget_gb,
            gpu_memory_budget_gb: config.gpu_memory_budget_gb,
            max_dense_train_particles: config.max_dense_train_particles,
            max_dense_chunk_floats: config.max_dense_chunk_floats,
            max_splat_chunk_floats: config.max_splat_chunk_floats,
        }
    }

    const MAX_QUALITY_EVAL_PARTICLE_ROWS: usize = 32 * 1024;

    pub(super) fn bounded_quality_eval_batch_size(
        requested_examples: usize,
        particle_count: usize,
    ) -> usize {
        requested_examples
            .max(1)
            .min(
                MAX_QUALITY_EVAL_PARTICLE_ROWS
                    .checked_div(particle_count.max(1))
                    .unwrap_or(1)
                    .max(1),
            )
    }

    pub(super) fn validation_direct_config(config: BurnE2eRolloutTrainConfig) -> DirectBasisTrainConfig {
        let mut direct = direct_config_view(config);
        direct.rollout_particles = config.validation_particles.max(1);
        direct.rollout_step_min = config.validation_steps.max(1);
        direct.rollout_steps = config.validation_steps.max(1);
        direct.update_prob = config.validation_update_prob;
        direct.seed = config.validation_seed;
        direct.eval_batch_size = if direct.rollout_particles > config.max_dense_train_particles {
            1
        } else {
            bounded_quality_eval_batch_size(
                config.example_batch_size,
                direct.rollout_particles,
            )
        };
        direct
    }

    impl BurnE2eConditionCache {
        #[cfg_attr(not(feature = "dino"), allow(unused_variables))]
        pub(super) fn from_examples_drain(
            examples: &mut [BurnE2eRolloutExample],
            device: &BurnDevice,
            device_cache_max_bytes: usize,
            config: BurnE2eRolloutTrainConfig,
        ) -> AutomataResult<Self> {
            if examples.is_empty() {
                return Ok(Self {
                    values: BurnE2eConditionValues::HostRows(Vec::new()),
                    teacher_vectors: None,
                    examples: 0,
                    token_count: 0,
                    embed_dims: 0,
                    device: device.clone(),
                });
            }
            let first = &examples[0];
            let token_count = first.token_count;
            let embed_dims = first.embed_dims;
            if token_count == 0 || embed_dims == 0 {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA e2e condition cache requires non-empty token shapes".to_string(),
                ));
            }
            let row_len = token_count * embed_dims;
            let teacher_vectors = if examples.iter().all(|example| example.teacher_adapter.is_some()) {
                let teacher_len = examples[0]
                    .teacher_adapter
                    .as_ref()
                    .map_or(0, Vec::len);
                if teacher_len == 0
                    || examples.iter().any(|example| {
                        example.teacher_adapter.as_ref().map(Vec::len) != Some(teacher_len)
                    })
                {
                    return Err(AutomataError::InvalidArgument(
                        "HyperNPA teacher adapter vectors must have one homogeneous non-empty shape"
                            .to_string(),
                    ));
                }
                Some(tensor(
                    examples
                        .iter()
                        .flat_map(|example| example.teacher_adapter.as_ref().unwrap().iter().copied())
                        .collect(),
                    [examples.len(), teacher_len],
                    device,
                ))
            } else if examples.iter().any(|example| example.teacher_adapter.is_some()) {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA examples must either all or none provide teacher adapters".to_string(),
                ));
            } else {
                None
            };
            let feature_bytes = examples
                .len()
                .saturating_mul(row_len)
                .saturating_mul(std::mem::size_of::<f32>());
            let static_rows = examples
                .iter()
                .all(|example| example.condition_features.len() == row_len);
            let dynamic_dino_rows = examples
                .iter()
                .all(|example| example.condition_features.is_empty() && example.condition_path.is_some());
            if !static_rows && !dynamic_dino_rows {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA e2e condition examples must be all static feature rows or all DINO image paths".to_string(),
                ));
            }
            if dynamic_dino_rows {
                #[cfg(feature = "dino")]
                {
                    let model_path = first.dino_model_path.as_ref().ok_or_else(|| {
                        AutomataError::InvalidArgument(
                            "DINO on-demand condition source requires condition.dino_model"
                                .to_string(),
                        )
                    })?;
                    let encoder =
                        DinoVitsConditionEncoderBackend::<InnerBackend>::load(
                            model_path,
                            config.dino_image_size,
                        )
                        .map_err(|err| {
                            AutomataError::InvalidArgument(format!(
                                "failed to load DINO model {}: {err}",
                                model_path.display()
                            ))
                        })?;
                    let paths = examples
                        .iter()
                        .map(|example| {
                            example.condition_path.clone().ok_or_else(|| {
                                AutomataError::InvalidArgument(
                                    "DINO condition example is missing condition_path".to_string(),
                                )
                            })
                        })
                        .collect::<AutomataResult<Vec<_>>>()?;
                    let token_cache_rows = device_cache_max_bytes
                        .checked_div(
                            token_count
                                .saturating_mul(embed_dims)
                                .saturating_mul(std::mem::size_of::<f32>())
                                .max(1),
                        )
                        .unwrap_or(0)
                        .min(examples.len());
                    let source = BurnE2eDinoConditionSource {
                        paths,
                        encoder,
                        batch_size: config.dino_batch_size.max(1),
                        token_grid_width: config.dino_token_grid_width,
                        token_grid_height: config.dino_token_grid_height,
                        l2_normalize_features: config.dino_l2_normalize_features,
                        rgb_channels: config.dino_rgb_channels,
                        rgb_channel_scale: config.dino_rgb_channel_scale,
                        alpha_channel: config.dino_alpha_channel,
                        alpha_channel_scale: config.dino_alpha_channel_scale,
                        patch_pixels: config.dino_patch_pixels,
                        token_cache: Mutex::new(BurnE2eDinoTokenCache {
                            values: None,
                            slot_identities: vec![None; token_cache_rows],
                            identity_slots: HashMap::with_capacity(token_cache_rows),
                            next_evict: 0,
                            hits: 0,
                            misses: 0,
                            inserts: 0,
                            evictions: 0,
                        }),
                    };
                    let values = if device_cache_max_bytes > 0
                        && feature_bytes <= device_cache_max_bytes
                    {
                        eprintln!(
                            "encoding {} DINO conditions into a bounded {:.2} GiB device token cache",
                            examples.len(),
                            feature_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                        );
                        BurnE2eConditionValues::Device(source.encode_all_to_device(
                            token_count,
                            embed_dims,
                            device,
                        )?)
                    } else {
                        BurnE2eConditionValues::DynamicDino(Box::new(source))
                    };
                    return Ok(Self {
                        values,
                        teacher_vectors,
                        examples: examples.len(),
                        token_count,
                        embed_dims,
                        device: device.clone(),
                    });
                }
                #[cfg(not(feature = "dino"))]
                {
                    return Err(AutomataError::InvalidArgument(
                        "DINO on-demand condition source requires the dino feature".to_string(),
                    ));
                }
            }
            let use_device_cache =
                device_cache_max_bytes > 0 && feature_bytes <= device_cache_max_bytes;
            let mut flat_values = use_device_cache
                .then(|| Vec::with_capacity(examples.len().saturating_mul(row_len)));
            let mut rows = (!use_device_cache).then(|| Vec::with_capacity(examples.len()));
            for example in &mut *examples {
                let feature_len = example.condition_features.len();
                if example.token_count != token_count
                    || example.embed_dims != embed_dims
                    || feature_len != row_len
                {
                    return Err(AutomataError::InvalidArgument(
                        "condition token shape mismatch in HyperNPA e2e cache".to_string(),
                    ));
                }
                let condition_features = std::mem::take(&mut example.condition_features);
                if let Some(flat_values) = flat_values.as_mut() {
                    flat_values.extend(condition_features);
                } else if let Some(rows) = rows.as_mut() {
                    rows.push(condition_features);
                }
            }
            let values = if let Some(values) = flat_values {
                BurnE2eConditionValues::Device(tensor3(
                    values,
                    [examples.len(), token_count, embed_dims],
                    device,
                ))
            } else {
                BurnE2eConditionValues::HostRows(rows.unwrap_or_default())
            };
            Ok(Self {
                values,
                teacher_vectors,
                examples: examples.len(),
                token_count,
                embed_dims,
                device: device.clone(),
            })
        }

        pub(super) fn select_teacher(&self, indices: &[usize]) -> Option<Tensor2> {
            self.teacher_vectors.as_ref().map(|teachers| {
                teachers.clone().select(
                    0,
                    Tensor::<BurnBackend, 1, Int>::from_data(
                        TensorData::new(
                            indices.iter().map(|index| *index as i64).collect::<Vec<_>>(),
                            [indices.len()],
                        ),
                        &self.device,
                    ),
                )
            })
        }

        pub(super) fn mean_pairwise_l2(&self) -> AutomataResult<Option<f32>> {
            if self.examples < 2 {
                return Ok(None);
            }
            let indices = condition_diagnostic_indices(self.examples);
            let values = tensor3_vec(self.select(&indices)?.inner())?;
            let row_len = self.token_count * self.embed_dims;
            let mut sum = 0.0_f64;
            let mut pairs = 0usize;
            for lhs in 0..indices.len() {
                for rhs in lhs + 1..indices.len() {
                    let lhs = &values[lhs * row_len..(lhs + 1) * row_len];
                    let rhs = &values[rhs * row_len..(rhs + 1) * row_len];
                    let distance = lhs
                        .iter()
                        .zip(rhs)
                        .map(|(lhs, rhs)| {
                            let delta = f64::from(*lhs - *rhs);
                            delta * delta
                        })
                        .sum::<f64>()
                        .sqrt();
                    sum += distance;
                    pairs += 1;
                }
            }
            Ok(Some((sum / pairs.max(1) as f64) as f32))
        }

        pub(super) fn nearest_rows(
            &self,
            queries: &Self,
            query_indices: &[usize],
        ) -> AutomataResult<Vec<(usize, f32)>> {
            if self.examples == 0 {
                return Err(AutomataError::InvalidArgument(
                    "nearest-condition lookup requires non-empty reference conditions".to_string(),
                ));
            }
            if self.token_count != queries.token_count || self.embed_dims != queries.embed_dims {
                return Err(AutomataError::InvalidArgument(
                    "nearest-condition lookup requires matching token shapes".to_string(),
                ));
            }
            let reference_indices = (0..self.examples).collect::<Vec<_>>();
            let references = tensor3_vec(self.select(&reference_indices)?.inner())?;
            let query_values = tensor3_vec(queries.select(query_indices)?.inner())?;
            let row_len = self.token_count * self.embed_dims;
            Ok(query_values
                .chunks_exact(row_len)
                .map(|query| {
                    references
                        .chunks_exact(row_len)
                        .enumerate()
                        .map(|(idx, reference)| {
                            let squared = query
                                .iter()
                                .zip(reference)
                                .map(|(query, reference)| {
                                    let delta = f64::from(*query - *reference);
                                    delta * delta
                                })
                                .sum::<f64>();
                            (idx, squared)
                        })
                        .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
                        .map(|(idx, squared)| (idx, squared.sqrt() as f32))
                        .expect("non-empty nearest-condition reference set")
                })
                .collect())
        }

        pub(super) fn mean_teacher_pairwise_l2(&self) -> AutomataResult<Option<f32>> {
            if self.teacher_vectors.is_none() {
                return Ok(None);
            }
            if self.examples < 2 {
                return Ok(None);
            }
            let indices = condition_diagnostic_indices(self.examples);
            let teachers = self
                .select_teacher(&indices)
                .expect("teacher diagnostics require teacher vectors");
            let dims = teachers.shape().dims::<2>();
            let values = tensor_vec(teachers.inner())?;
            let mut sum = 0.0_f64;
            let mut pairs = 0usize;
            for lhs in 0..indices.len() {
                for rhs in lhs + 1..indices.len() {
                    let lhs = &values[lhs * dims[1]..(lhs + 1) * dims[1]];
                    let rhs = &values[rhs * dims[1]..(rhs + 1) * dims[1]];
                    sum += lhs
                        .iter()
                        .zip(rhs)
                        .map(|(lhs, rhs)| {
                            let delta = f64::from(*lhs - *rhs);
                            delta * delta
                        })
                        .sum::<f64>()
                        .sqrt();
                    pairs += 1;
                }
            }
            Ok(Some((sum / pairs.max(1) as f64) as f32))
        }

        pub(super) fn select(&self, indices: &[usize]) -> AutomataResult<Tensor3> {
            if indices.is_empty() || self.token_count == 0 || self.embed_dims == 0 {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA e2e condition cache select requires non-empty indices".to_string(),
                ));
            }
            if indices.iter().any(|idx| *idx >= self.examples) {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA e2e condition cache index out of bounds".to_string(),
                ));
            }
            match &self.values {
                BurnE2eConditionValues::Device(values) => {
                    let index_values = indices.iter().map(|idx| *idx as i64).collect::<Vec<_>>();
                    let index_tensor: Tensor1Int =
                        Tensor::from_data(TensorData::new(index_values, [indices.len()]), &self.device);
                    Ok(values.clone().select(0, index_tensor))
                }
                BurnE2eConditionValues::HostRows(rows) => {
                    let row_len = self.token_count * self.embed_dims;
                    let mut selected = Vec::with_capacity(indices.len() * row_len);
                    for &idx in indices {
                        selected.extend_from_slice(&rows[idx]);
                    }
                    Ok(tensor3(
                        selected,
                        [indices.len(), self.token_count, self.embed_dims],
                        &self.device,
                    ))
                }
                #[cfg(feature = "dino")]
                BurnE2eConditionValues::DynamicDino(source) => {
                    source.select(indices, self.token_count, self.embed_dims)
                }
            }
        }

        #[cfg_attr(not(feature = "dino"), allow(unused_variables))]
        pub(super) fn select_prepared(
            &self,
            indices: &[usize],
            prepared_dino: Option<&BurnE2ePreparedDinoBatch>,
        ) -> AutomataResult<Tensor3> {
            #[cfg(feature = "dino")]
            if let (BurnE2eConditionValues::DynamicDino(source), Some(prepared)) =
                (&self.values, prepared_dino)
            {
                let (unique_indices, expected_expansion) =
                    deduplicate_condition_indices(indices);
                if prepared.expansion != expected_expansion
                    || prepared.encoded_rows != unique_indices.len()
                    || prepared.encoded_identities != unique_indices
                {
                    return Err(AutomataError::InvalidArgument(
                        "prepared DINO batch must contain every unique requested condition row"
                            .to_string(),
                    ));
                }
                if let Some(cached) = source.select_cached(indices)? {
                    return Ok(cached);
                }
                let encoded_identities =
                    if source.token_cache_capacity() >= unique_indices.len() {
                        source.missing_identities(&unique_indices)?
                    } else {
                        unique_indices.clone()
                    };
                let prepared_rows = encoded_identities
                    .iter()
                    .map(|identity| {
                        unique_indices
                            .iter()
                            .position(|candidate| candidate == identity)
                            .expect("missing DINO identity must be present in the prepared batch")
                    })
                    .collect::<Vec<_>>();
                let selected = prepared
                    .prepared
                    .select_rows(&prepared_rows)
                    .map_err(|error| {
                        AutomataError::InvalidArgument(format!(
                            "failed to select prepared DINO condition rows: {error}"
                        ))
                    })?;
                let encoded = source.encode_preprocessed(
                    &selected,
                    encoded_identities.len(),
                    self.token_count,
                    self.embed_dims,
                )?;
                source.cache_encoded(&encoded_identities, encoded.clone(), &unique_indices)?;
                if encoded_identities == unique_indices
                    && unique_indices.len() == indices.len()
                    && expected_expansion.iter().copied().eq(0..indices.len())
                {
                    return Ok(encoded);
                }
                if encoded_identities == unique_indices {
                    let device = encoded.device();
                    return Ok(encoded.select(
                        0,
                        Tensor::<BurnBackend, 1, Int>::from_data(
                            TensorData::new(
                                expected_expansion
                                    .iter()
                                    .map(|row| *row as i64)
                                    .collect::<Vec<_>>(),
                                [expected_expansion.len()],
                            ),
                            &device,
                        ),
                    ));
                }
                return source.gather_cached(indices)?.ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "partial prepared DINO batch did not populate every requested cache row"
                            .to_string(),
                    )
                });
            }
            self.select(indices)
        }

        pub(super) fn dynamic_dino_paths_for_indices(
            &self,
            indices: &[usize],
        ) -> AutomataResult<Option<(Vec<usize>, Vec<PathBuf>)>> {
            if indices.iter().any(|idx| *idx >= self.examples) {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA e2e condition cache index out of bounds".to_string(),
                ));
            }
            #[cfg(feature = "dino")]
            if let BurnE2eConditionValues::DynamicDino(source) = &self.values {
                let encoded_identities = indices.to_vec();
                if encoded_identities.is_empty() {
                    return Ok(None);
                }
                let paths = encoded_identities
                    .iter()
                    .map(|idx| {
                        source.paths.get(*idx).cloned().ok_or_else(|| {
                            AutomataError::InvalidArgument(
                                "DINO condition source index out of bounds".to_string(),
                            )
                        })
                    })
                    .collect::<AutomataResult<Vec<_>>>()?;
                return Ok(Some((encoded_identities, paths)));
            }
            Ok(None)
        }

        pub(super) fn feature_bytes(&self) -> usize {
            #[cfg(feature = "dino")]
            if let BurnE2eConditionValues::DynamicDino(source) = &self.values {
                return source.token_cache_bytes(self.token_count, self.embed_dims);
            }
            self.examples
                .saturating_mul(self.token_count)
                .saturating_mul(self.embed_dims)
                .saturating_mul(std::mem::size_of::<f32>())
        }

        pub(super) fn storage_label(&self) -> &'static str {
            if self.examples == 0 {
                return "empty";
            }
            match &self.values {
                BurnE2eConditionValues::Device(_) => "device-resident",
                BurnE2eConditionValues::HostRows(_) => "host-row-streamed",
                #[cfg(feature = "dino")]
                BurnE2eConditionValues::DynamicDino(source) => {
                    if source.token_cache_capacity() > 0 {
                        "dino-on-demand-device-lru"
                    } else {
                        "dino-on-demand-device"
                    }
                }
            }
        }

        pub(super) fn is_device_resident(&self) -> bool {
            matches!(self.values, BurnE2eConditionValues::Device(_))
        }

        pub(super) fn drained_cpu_features_from_examples(&self) -> bool {
            if self.examples == 0 {
                return false;
            }
            match &self.values {
                BurnE2eConditionValues::Device(_) | BurnE2eConditionValues::HostRows(_) => true,
                #[cfg(feature = "dino")]
                BurnE2eConditionValues::DynamicDino(_) => false,
            }
        }

        pub(super) fn token_cache_metrics(&self) -> serde_json::Value {
            #[cfg(feature = "dino")]
            if let BurnE2eConditionValues::DynamicDino(source) = &self.values {
                return source.token_cache_metrics();
            }
            json!({
                "mode": self.storage_label(),
                "capacity_rows": self.examples,
                "resident_rows": self.examples,
                "hits": 0,
                "misses": 0,
                "inserts": 0,
                "evictions": 0,
            })
        }
    }

    #[cfg(feature = "dino")]
    impl BurnE2eDinoConditionSource {
        pub(super) fn encode_all_to_device(
            &self,
            token_count: usize,
            embed_dims: usize,
            device: &BurnDevice,
        ) -> AutomataResult<Tensor3> {
            let mut values = Tensor::<InnerBackend, 3>::zeros(
                [self.paths.len(), token_count, embed_dims],
                device,
            );
            let batches = self.paths.len().div_ceil(self.batch_size);
            for (batch, paths) in self.paths.chunks(self.batch_size).enumerate() {
                let images = paths
                    .par_iter()
                    .map(|path| load_dino_condition_image(path))
                    .collect::<AutomataResult<Vec<_>>>()?;
                let encoded = self
                    .encode_loaded(&images, token_count, embed_dims)?
                    .inner();
                let start = batch * self.batch_size;
                let slots = (start..start + paths.len()).collect::<Vec<_>>();
                let slot_indices = inner_index_tensor(&slots, device);
                values = values.select_assign(
                    0,
                    slot_indices,
                    encoded,
                    IndexingUpdateOp::Add,
                );
                let completed = batch + 1;
                if completed == batches || completed.is_multiple_of(64) {
                    eprintln!("encoded DINO device cache batch {completed}/{batches}");
                }
            }
            Ok(Tensor::<BurnBackend, 3>::from_inner(values))
        }

        pub(super) fn select(
            &self,
            indices: &[usize],
            token_count: usize,
            embed_dims: usize,
        ) -> AutomataResult<Tensor3> {
            if let Some(cached) = self.select_cached(indices)? {
                return Ok(cached);
            }
            let (unique_indices, expansion) = deduplicate_condition_indices(indices);
            let encoded_identities = if self.token_cache_capacity() >= unique_indices.len() {
                self.missing_identities(&unique_indices)?
            } else {
                unique_indices.clone()
            };
            let mut chunks =
                Vec::with_capacity(encoded_identities.len().div_ceil(self.batch_size));
            for chunk_indices in encoded_identities.chunks(self.batch_size) {
                let conditions = chunk_indices
                    .iter()
                    .map(|idx| {
                        let path = self.paths.get(*idx).ok_or_else(|| {
                            AutomataError::InvalidArgument(
                                "DINO condition source index out of bounds".to_string(),
                            )
                        })?;
                        load_dino_condition_image(path)
                    })
                    .collect::<AutomataResult<Vec<_>>>()?;
                let encoded = self
                    .encoder
                    .encode_batch_tensor_with_contract(&conditions, self.contract())
                    .map_err(|err| {
                        AutomataError::InvalidArgument(format!(
                            "failed to encode on-demand DINO condition batch: {err}"
                        ))
                    })?;
                chunks.push(encoded);
            }
            let encoded = if chunks.len() == 1 {
                chunks.remove(0)
            } else {
                Tensor::cat(chunks, 0)
            };
            let dims = encoded.dims();
            if dims != [encoded_identities.len(), token_count, embed_dims] {
                return Err(AutomataError::InvalidArgument(format!(
                    "on-demand DINO condition tensor shape {:?} != [{}, {}, {}]",
                    dims,
                    encoded_identities.len(),
                    token_count,
                    embed_dims
                )));
            }
            let encoded = Tensor::<BurnBackend, 3>::from_inner(encoded);
            self.cache_encoded(&encoded_identities, encoded.clone(), &unique_indices)?;
            if encoded_identities != unique_indices {
                return self.gather_cached(indices)?.ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "partial on-demand DINO encoding did not populate every requested cache row"
                            .to_string(),
                    )
                });
            }
            if unique_indices.len() == indices.len()
                && expansion.iter().copied().eq(0..indices.len())
            {
                return Ok(encoded);
            }
            let device = encoded.device();
            Ok(encoded.select(
                0,
                Tensor::<BurnBackend, 1, Int>::from_data(
                    TensorData::new(
                        expansion
                            .into_iter()
                            .map(|row| row as i64)
                            .collect::<Vec<_>>(),
                        [indices.len()],
                    ),
                    &device,
                ),
            ))
        }

        pub(super) fn token_cache_capacity(&self) -> usize {
            self.token_cache
                .lock()
                .map(|cache| cache.slot_identities.len())
                .unwrap_or(0)
        }

        pub(super) fn token_cache_bytes(&self, token_count: usize, embed_dims: usize) -> usize {
            self.token_cache_capacity()
                .saturating_mul(token_count)
                .saturating_mul(embed_dims)
                .saturating_mul(std::mem::size_of::<f32>())
        }

        pub(super) fn token_cache_metrics(&self) -> serde_json::Value {
            match self.token_cache.lock() {
                Ok(cache) => {
                    let lookups = cache.hits.saturating_add(cache.misses);
                    json!({
                        "mode": "bounded-device-lru-plus-on-demand",
                        "capacity_rows": cache.slot_identities.len(),
                        "resident_rows": cache.identity_slots.len(),
                        "hits": cache.hits,
                        "misses": cache.misses,
                        "hit_rate": if lookups == 0 {
                            0.0
                        } else {
                            cache.hits as f64 / lookups as f64
                        },
                        "inserts": cache.inserts,
                        "evictions": cache.evictions,
                    })
                }
                Err(_) => json!({
                    "mode": "bounded-device-lru-plus-on-demand",
                    "error": "poisoned",
                }),
            }
        }

        pub(super) fn missing_identities(
            &self,
            indices: &[usize],
        ) -> AutomataResult<Vec<usize>> {
            let cache = self.token_cache.lock().map_err(|_| {
                AutomataError::InvalidArgument("DINO token cache lock is poisoned".to_string())
            })?;
            Ok(indices
                .iter()
                .copied()
                .filter(|identity| !cache.identity_slots.contains_key(identity))
                .collect())
        }

        fn select_cached(&self, indices: &[usize]) -> AutomataResult<Option<Tensor3>> {
            self.select_cached_impl(indices, true)
        }

        fn gather_cached(&self, indices: &[usize]) -> AutomataResult<Option<Tensor3>> {
            self.select_cached_impl(indices, false)
        }

        fn select_cached_impl(
            &self,
            indices: &[usize],
            record_lookup: bool,
        ) -> AutomataResult<Option<Tensor3>> {
            let mut cache = self.token_cache.lock().map_err(|_| {
                AutomataError::InvalidArgument("DINO token cache lock is poisoned".to_string())
            })?;
            if indices.is_empty() {
                return Ok(None);
            }
            let missing = indices
                .iter()
                .filter(|identity| !cache.identity_slots.contains_key(identity))
                .count();
            if record_lookup {
                cache.hits = cache
                    .hits
                    .saturating_add(indices.len().saturating_sub(missing) as u64);
                cache.misses = cache.misses.saturating_add(missing as u64);
            }
            if missing > 0 {
                return Ok(None);
            }
            let Some(values) = cache.values.as_ref() else {
                return Ok(None);
            };
            let slots = indices
                .iter()
                .map(|identity| cache.identity_slots[identity])
                .collect::<Vec<_>>();
            let device = values.device();
            Ok(Some(Tensor::<BurnBackend, 3>::from_inner(
                values
                    .clone()
                    .select(0, inner_index_tensor(&slots, &device)),
            )))
        }

        fn cache_encoded(
            &self,
            identities: &[usize],
            encoded: Tensor3,
            protected_identities: &[usize],
        ) -> AutomataResult<()> {
            if identities.is_empty() {
                return Ok(());
            }
            let dims = encoded.shape().dims::<3>();
            if dims[0] != identities.len() {
                return Err(AutomataError::InvalidArgument(format!(
                    "DINO cache insert has {} identities for {} encoded rows",
                    identities.len(),
                    dims[0],
                )));
            }
            let mut cache = self.token_cache.lock().map_err(|_| {
                AutomataError::InvalidArgument("DINO token cache lock is poisoned".to_string())
            })?;
            let capacity = cache.slot_identities.len();
            if capacity == 0 || identities.len() > capacity {
                return Ok(());
            }
            let protected = protected_identities.iter().copied().collect::<HashSet<_>>();
            if identities
                .iter()
                .any(|identity| !protected.contains(identity))
            {
                return Err(AutomataError::InvalidArgument(
                    "DINO cache insert identities must be included in the protected request"
                        .to_string(),
                ));
            }
            let mut slots = Vec::with_capacity(identities.len());
            for &identity in identities {
                if let Some(&slot) = cache.identity_slots.get(&identity) {
                    slots.push(slot);
                    continue;
                }
                let slot = {
                    let BurnE2eDinoTokenCache {
                        slot_identities,
                        next_evict,
                        ..
                    } = &mut *cache;
                    next_e2e_dino_cache_slot(slot_identities, &protected, next_evict)
                }
                .ok_or_else(|| {
                    AutomataError::InvalidArgument(
                        "DINO token cache cannot insert a requested row without evicting another requested row"
                            .to_string(),
                    )
                })?;
                if let Some(evicted) = cache.slot_identities[slot].take() {
                    cache.identity_slots.remove(&evicted);
                    cache.evictions = cache.evictions.saturating_add(1);
                }
                cache.slot_identities[slot] = Some(identity);
                cache.identity_slots.insert(identity, slot);
                cache.inserts = cache.inserts.saturating_add(1);
                slots.push(slot);
            }

            let device = encoded.device();
            let mut values = cache.values.take().unwrap_or_else(|| {
                Tensor::<InnerBackend, 3>::zeros([capacity, dims[1], dims[2]], &device)
            });
            let slot_indices = inner_index_tensor(&slots, &device);
            let encoded = encoded.inner();
            let delta = encoded - values.clone().select(0, slot_indices.clone());
            values = values.select_assign(0, slot_indices, delta, IndexingUpdateOp::Add);
            cache.values = Some(values);
            Ok(())
        }

        pub(super) fn encode_preprocessed(
            &self,
            prepared: &DinoVitsPreparedConditionBatch,
            batch: usize,
            token_count: usize,
            embed_dims: usize,
        ) -> AutomataResult<Tensor3> {
            if batch == 0 {
                return Err(AutomataError::InvalidArgument(
                    "preprocessed DINO condition batch is empty".to_string(),
                ));
            }
            let encoded = self
                .encoder
                .encode_preprocessed_batch_tensor_with_contract(prepared, self.contract())
                .map_err(|err| {
                    AutomataError::InvalidArgument(format!(
                        "failed to encode preprocessed DINO condition batch: {err}"
                    ))
                })?;
            let dims = encoded.dims();
            if dims != [batch, token_count, embed_dims] {
                return Err(AutomataError::InvalidArgument(format!(
                    "preprocessed DINO condition tensor shape {:?} != [{}, {}, {}]",
                    dims, batch, token_count, embed_dims
                )));
            }
            Ok(Tensor::<BurnBackend, 3>::from_inner(encoded))
        }

        pub(super) fn encode_loaded(
            &self,
            images: &[ConditionImage2d],
            token_count: usize,
            embed_dims: usize,
        ) -> AutomataResult<Tensor3> {
            let mut chunks = Vec::with_capacity(images.len().div_ceil(self.batch_size));
            for conditions in images.chunks(self.batch_size) {
                let encoded = self
                    .encoder
                    .encode_batch_tensor_with_contract(conditions, self.contract())
                    .map_err(|err| {
                        AutomataError::InvalidArgument(format!(
                            "failed to encode preloaded DINO condition batch: {err}"
                        ))
                    })?;
                chunks.push(encoded);
            }
            let encoded = if chunks.len() == 1 {
                chunks.remove(0)
            } else {
                Tensor::cat(chunks, 0)
            };
            let dims = encoded.dims();
            if dims != [images.len(), token_count, embed_dims] {
                return Err(AutomataError::InvalidArgument(format!(
                    "preloaded DINO condition tensor shape {:?} != [{}, {}, {}]",
                    dims,
                    images.len(),
                    token_count,
                    embed_dims
                )));
            }
            Ok(Tensor::<BurnBackend, 3>::from_inner(encoded))
        }
    }

    #[cfg(feature = "dino")]
    pub(super) fn load_dino_condition_image(path: &Path) -> AutomataResult<ConditionImage2d> {
        crate::load_condition_image(path).map_err(|err| {
            AutomataError::InvalidArgument(format!(
                "failed to load condition image {}: {err}",
                path.display()
            ))
        })
    }

    pub(super) fn seed_batch_tensors(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        config: DirectBasisTrainConfig,
        step_seed: u64,
        device: &BurnDevice,
    ) -> (Tensor3, Tensor3) {
        seed_batch_tensors_with_seed_indices(
            targets,
            indices,
            indices,
            particle_count,
            config,
            step_seed,
            device,
        )
    }

    pub(super) fn seed_batch_tensors_with_seed_indices(
        targets: &[BurnTargetExample],
        target_indices: &[usize],
        seed_indices: &[usize],
        particle_count: usize,
        config: DirectBasisTrainConfig,
        step_seed: u64,
        device: &BurnDevice,
    ) -> (Tensor3, Tensor3) {
        debug_assert_eq!(target_indices.len(), seed_indices.len());
        let mut positions = Vec::with_capacity(target_indices.len() * particle_count * 2);
        let mut states = Vec::with_capacity(target_indices.len() * particle_count * 16);
        for (&target_idx, &seed_idx) in target_indices.iter().zip(seed_indices) {
            let (example_positions, example_states) = seed_particles_scaled(
                1,
                particle_count,
                16,
                2,
                step_seed.wrapping_add(seed_idx as u64),
                config.seed_mode,
                targets[target_idx].seed_scale,
            );
            positions.extend(
                example_positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]]),
            );
            states.extend(example_states);
        }
        (
            tensor3(positions, [target_indices.len(), particle_count, 2], device),
            tensor3(states, [target_indices.len(), particle_count, 16], device),
        )
    }

    pub(super) fn host_batch_mask(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        rng: &mut StdRng,
    ) -> Tensor3 {
        let mut values = Vec::with_capacity(indices.len() * particle_count);
        for &idx in indices {
            values.extend(stochastic_mask(
                particle_count,
                targets[idx].update_prob,
                rng,
            ));
        }
        STOCHASTIC_MASK_UPLOAD_HITS.fetch_add(1, Ordering::Relaxed);
        tensor3(
            values,
            [indices.len(), particle_count, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    pub(super) fn host_single_mask_stack(
        target: &BurnTargetExample,
        steps: usize,
        rng: &mut StdRng,
    ) -> Tensor3 {
        let mut values = Vec::with_capacity(steps * target.particle_count);
        for _ in 0..steps {
            values.extend(stochastic_mask(
                target.particle_count,
                target.update_prob,
                rng,
            ));
        }
        STOCHASTIC_MASK_UPLOAD_HITS.fetch_add(1, Ordering::Relaxed);
        tensor3(
            values,
            [steps, target.particle_count, 1],
            &target.target_rgb.device(),
        )
    }

    pub(super) fn host_batch_mask_stack(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        steps: usize,
        rng: &mut StdRng,
    ) -> Tensor4 {
        let mut values = Vec::with_capacity(steps * indices.len() * particle_count);
        for _ in 0..steps {
            for &idx in indices {
                values.extend(stochastic_mask(
                    particle_count,
                    targets[idx].update_prob,
                    rng,
                ));
            }
        }
        STOCHASTIC_MASK_UPLOAD_HITS.fetch_add(1, Ordering::Relaxed);
        tensor4(
            values,
            [steps, indices.len(), particle_count, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    pub(super) fn device_batch_mask_stack(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        steps: usize,
    ) -> Tensor4 {
        let device = &targets[indices[0]].target_rgb.device();
        let shape = [steps, indices.len(), particle_count, 1];
        let samples = Tensor::<BurnBackend, 4>::random(
            shape,
            Distribution::Uniform(0.0, 1.0),
            device,
        );
        STOCHASTIC_MASK_DEVICE_HITS.fetch_add(1, Ordering::Relaxed);
        if let Some(update_prob) = homogeneous_update_prob(targets, indices) {
            return samples.lower_elem(update_prob).float();
        }
        let probs = indices
            .iter()
            .map(|idx| targets[*idx].update_prob)
            .collect::<Vec<_>>();
        let probs = tensor4(probs, [1, indices.len(), 1, 1], device).expand(shape);
        samples.lower(probs).float()
    }

    pub(super) fn batch_update_prob_is_one(targets: &[BurnTargetExample], indices: &[usize]) -> bool {
        indices.iter().all(|&idx| targets[idx].update_prob >= 1.0)
    }

    pub(super) fn homogeneous_update_prob(targets: &[BurnTargetExample], indices: &[usize]) -> Option<f32> {
        let first = targets[*indices.first()?].update_prob;
        indices
            .iter()
            .all(|&idx| (targets[idx].update_prob - first).abs() <= f32::EPSILON)
            .then_some(first)
    }

    pub(super) fn host_batch_mask_with_rngs(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        rngs: &mut [StdRng],
    ) -> Tensor3 {
        let mut values = Vec::with_capacity(indices.len() * particle_count);
        for (local, &idx) in indices.iter().enumerate() {
            values.extend(stochastic_mask(
                particle_count,
                targets[idx].update_prob,
                &mut rngs[local],
            ));
        }
        STOCHASTIC_MASK_UPLOAD_HITS.fetch_add(1, Ordering::Relaxed);
        tensor3(
            values,
            [indices.len(), particle_count, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    pub(super) fn host_batch_mask_stack_with_rngs(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        steps: usize,
        rngs: &mut [StdRng],
    ) -> Tensor4 {
        let mut values = Vec::with_capacity(steps * indices.len() * particle_count);
        for _ in 0..steps {
            for (local, &idx) in indices.iter().enumerate() {
                values.extend(stochastic_mask(
                    particle_count,
                    targets[idx].update_prob,
                    &mut rngs[local],
                ));
            }
        }
        STOCHASTIC_MASK_UPLOAD_HITS.fetch_add(1, Ordering::Relaxed);
        tensor4(
            values,
            [steps, indices.len(), particle_count, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    pub(super) fn host_batch_wgpu_mask_stack(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        steps: usize,
        step_offset: usize,
        seeds: &[u64],
        material_update_masks: Option<&[crate::adaptive::AdaptiveTarget2dUpdateMask]>,
    ) -> Tensor4 {
        debug_assert_eq!(indices.len(), seeds.len());
        debug_assert!(
            material_update_masks
                .is_none_or(|masks| masks.len() == indices.len() * particle_count)
        );
        let mut values = Vec::with_capacity(steps * indices.len() * particle_count);
        for step in 0..steps {
            let absolute_step = step_offset.saturating_add(step) as u32;
            for (local, &idx) in indices.iter().enumerate() {
                let probability = targets[idx].update_prob;
                let seed = wgpu_random_seed(seeds[local]);
                values.extend((0..particle_count).map(|particle| {
                    material_update_masks.map_or_else(
                        || {
                            f32::from(
                                wgpu_random01(particle as u32, absolute_step, seed) < probability,
                            )
                        },
                        |masks| {
                            let mask = masks[local * particle_count + particle];
                            if mask.expected {
                                probability
                            } else {
                                mask.keys
                                    .iter()
                                    .copied()
                                    .zip(mask.weights.iter().copied())
                                    .take_while(|(_, weight)| *weight > 0.0)
                                    .map(|(key, weight)| {
                                        weight
                                            * f32::from(
                                                wgpu_random01(key, absolute_step, seed)
                                                    < probability,
                                            )
                                    })
                                    .sum()
                            }
                        },
                    )
                }));
            }
        }
        STOCHASTIC_MASK_UPLOAD_HITS.fetch_add(1, Ordering::Relaxed);
        tensor4(
            values,
            [steps, indices.len(), particle_count, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    pub(super) fn wgpu_random_seed(seed: u64) -> u32 {
        (seed as u32) ^ ((seed >> 32) as u32)
    }

    fn wgpu_hash_u32(mut value: u32) -> u32 {
        value = (value ^ 61) ^ (value >> 16);
        value = value.wrapping_add(value << 3);
        value ^= value >> 4;
        value = value.wrapping_mul(0x27d4_eb2d);
        value ^ (value >> 15)
    }

    pub(super) fn wgpu_random01(particle: u32, step: u32, seed: u32) -> f32 {
        let mixed =
            wgpu_hash_u32(particle ^ wgpu_hash_u32(step.wrapping_add(0x9e37_79b9)) ^ seed);
        (mixed >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    pub(super) fn host_batch_mask_seeded(
        targets: &[BurnTargetExample],
        indices: &[usize],
        particle_count: usize,
        seed: u64,
    ) -> Tensor3 {
        let mut rng = StdRng::seed_from_u64(seed);
        host_batch_mask(targets, indices, particle_count, &mut rng)
    }

    pub(super) fn stack_target_rgb(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        Tensor::cat(
            indices
                .iter()
                .map(|idx| targets[*idx].target_rgb.clone().unsqueeze_dim::<3>(0))
                .collect::<Vec<_>>(),
            0,
        )
    }

    pub(super) fn stack_target_density(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        Tensor::cat(
            indices
                .iter()
                .map(|idx| targets[*idx].target_density.clone().unsqueeze_dim::<3>(0))
                .collect::<Vec<_>>(),
            0,
        )
    }

    pub(super) fn stack_target_foreground(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        Tensor::cat(
            indices
                .iter()
                .map(|idx| targets[*idx].target_foreground.clone().unsqueeze_dim::<3>(0))
                .collect::<Vec<_>>(),
            0,
        )
    }

    pub(super) fn stack_target_foreground_scales(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        let values = indices
            .iter()
            .map(|idx| targets[*idx].target_foreground_scale)
            .collect::<Vec<_>>();
        tensor3(
            values,
            [indices.len(), 1, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    pub(super) fn stack_target_mean(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        Tensor::cat(
            indices
                .iter()
                .map(|idx| targets[*idx].target_mean.clone().unsqueeze_dim::<3>(0))
                .collect::<Vec<_>>(),
            0,
        )
    }

    pub(super) fn stack_pixel_sizes(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        let values = indices
            .iter()
            .map(|idx| targets[*idx].pixel_size)
            .collect::<Vec<_>>();
        tensor3(
            values,
            [indices.len(), 1, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    pub(super) fn stack_target_point_counts(targets: &[BurnTargetExample], indices: &[usize]) -> Tensor3 {
        let values = indices
            .iter()
            .map(|idx| targets[*idx].target_points as f32)
            .collect::<Vec<_>>();
        tensor3(
            values,
            [indices.len(), 1, 1],
            &targets[indices[0]].target_rgb.device(),
        )
    }

    pub(super) fn pixel_xy_values(image_size: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(image_size * image_size * 2);
        for y in 0..image_size {
            for x in 0..image_size {
                values.push(x as f32);
                values.push(y as f32);
            }
        }
        values
    }

    pub(super) fn condition_patch_centers_values(width: usize, height: usize) -> Vec<f32> {
        let width = width.max(1);
        let height = height.max(1);
        let mut values = Vec::with_capacity(width * height * 2);
        for y in 0..height {
            let yy = ((y as f32 + 0.5) / height as f32) * 2.0 - 1.0;
            for x in 0..width {
                let xx = ((x as f32 + 0.5) / width as f32) * 2.0 - 1.0;
                values.push(xx);
                values.push(yy);
            }
        }
        values
    }

    pub(super) fn adapter_cache_metrics(
        base: &NpaModel,
        params: &BurnBaseParams,
        train_adapters: &[BurnAdapterParams],
        holdout_adapters: &[BurnAdapterParams],
        train_targets: &[BurnTargetExample],
        holdout_targets: &[BurnTargetExample],
    ) -> AutomataResult<serde_json::Value> {
        let rank = train_adapters
            .first()
            .or_else(|| holdout_adapters.first())
            .map_or(0, |adapter| adapter.rank);
        let parameters_per_adapter = if rank == 0 {
            0
        } else {
            NpaLowRankAdapter::parameter_count_for_config(&base.config, rank)
        };
        let total_adapters = train_adapters.len() + holdout_adapters.len();
        let total_adapter_parameters = parameters_per_adapter * total_adapters;
        let train_target_points = train_targets
            .iter()
            .map(|target| target.target_points)
            .sum::<usize>();
        let holdout_target_points = holdout_targets
            .iter()
            .map(|target| target.target_points)
            .sum::<usize>();
        let train_render_pixels = train_targets
            .iter()
            .map(|target| target.target_density.shape().dims::<2>()[0])
            .sum::<usize>();
        let holdout_render_pixels = holdout_targets
            .iter()
            .map(|target| target.target_density.shape().dims::<2>()[0])
            .sum::<usize>();
        Ok(json!({
            "representation": "resident_gpu_tensor_set_per_sample",
            "readback_policy": "report_interval_scalars_and_end_of_phase_artifacts_only",
            "non_report_step_loss_readbacks": false,
            "adapter_tensors_per_sample": 6,
            "rank": rank,
            "parameters_per_adapter": parameters_per_adapter,
            "train_adapters": train_adapters.len(),
            "holdout_adapters": holdout_adapters.len(),
            "total_adapters": total_adapters,
            "total_adapter_parameters": total_adapter_parameters,
            "estimated_adapter_weight_bytes_f32": total_adapter_parameters * std::mem::size_of::<f32>(),
            "estimated_adapter_tensor_count": total_adapters * 6,
            "train_target_points": train_target_points,
            "holdout_target_points": holdout_target_points,
            "train_render_pixels": train_render_pixels,
            "holdout_render_pixels": holdout_render_pixels,
            "estimated_target_render_cache_bytes_f32": (train_render_pixels + holdout_render_pixels) * 4 * std::mem::size_of::<f32>(),
            "base_norms": base_norm_metrics(params)?,
            "train_adapter_norms": adapter_norm_metrics(train_adapters)?,
            "holdout_adapter_norms": adapter_norm_metrics(holdout_adapters)?,
        }))
    }

    pub(super) fn base_norm_metrics(params: &BurnBaseParams) -> AutomataResult<serde_json::Value> {
        let w1 = tensor_l2_norm(&params.w1.clone().inner())?;
        let b1 = tensor_l2_norm(&params.b1.clone().inner())?;
        let w2 = tensor_l2_norm(&params.w2.clone().inner())?;
        let b2 = tensor_l2_norm(&params.b2.clone().inner())?;
        Ok(json!({
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "total": finite_scalar("Burn direct base norm", (w1 * w1 + b1 * b1 + w2 * w2 + b2 * b2).sqrt())?,
        }))
    }

    pub(super) fn adapter_norm_metrics(adapters: &[BurnAdapterParams]) -> AutomataResult<serde_json::Value> {
        if adapters.is_empty() {
            return Ok(json!({
                "examples": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
            }));
        }
        let mut sum = 0.0_f32;
        let mut min = f32::INFINITY;
        let mut max = 0.0_f32;
        for adapter in adapters {
            let norm = adapter_l2_norm(adapter)?;
            sum += norm;
            min = min.min(norm);
            max = max.max(norm);
        }
        Ok(json!({
            "examples": adapters.len(),
            "mean": finite_scalar("Burn direct mean adapter norm", sum / adapters.len() as f32)?,
            "min": finite_scalar("Burn direct min adapter norm", min)?,
            "max": finite_scalar("Burn direct max adapter norm", max)?,
        }))
    }

    pub(super) fn adapter_l2_norm(adapter: &BurnAdapterParams) -> AutomataResult<f32> {
        let tensors = [
            adapter.w1_down.clone().inner(),
            adapter.w1_up.clone().inner(),
            adapter.w2_down.clone().inner(),
            adapter.w2_up.clone().inner(),
            adapter.b1_delta.clone().inner(),
            adapter.b2_delta.clone().inner(),
        ];
        finite_scalar(
            "Burn direct adapter norm",
            group_norm_tensor(&tensors).into_scalar(),
        )
    }

    pub(super) fn mean_updates_per_sample(steps: usize, batch_size: usize, examples: usize) -> f32 {
        if examples == 0 {
            return 0.0;
        }
        steps as f32 * batch_size.min(examples).max(1) as f32 / examples as f32
    }

    pub(super) fn normalized_batch_size(requested: usize, examples: usize) -> usize {
        requested.max(1).min(examples.max(1))
    }

    pub(super) fn sample_indices(examples: usize, batch_size: usize, rng: &mut StdRng) -> Vec<usize> {
        let batch_size = batch_size.min(examples);
        if batch_size == 0 {
            return Vec::new();
        }
        if batch_size.saturating_mul(4) > examples {
            let mut indices = (0..examples).collect::<Vec<_>>();
            indices.shuffle(rng);
            indices.truncate(batch_size);
            return indices;
        }
        let mut indices = Vec::with_capacity(batch_size);
        while indices.len() < batch_size {
            let idx = rng.random_range(0..examples);
            if !indices.contains(&idx) {
                indices.push(idx);
            }
        }
        indices
    }

    pub(super) fn sample_rollout_indices(
        sampler: &mut E2eIdentitySampler,
        rollout_replicas: usize,
        rng: &mut StdRng,
    ) -> Vec<usize> {
        sampler
            .next_batch(rng)
            .into_iter()
            .flat_map(|example| std::iter::repeat_n(example, rollout_replicas.max(1)))
            .collect()
    }

    pub(super) fn e2e_sampling_rng(seed: u64, step: usize) -> StdRng {
        StdRng::seed_from_u64(
            seed ^ 0x51a9_1e5a_d00d_f00d_u64
                ^ (step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        )
    }

    pub(super) fn e2e_pool_rng(seed: u64, step: usize) -> StdRng {
        StdRng::seed_from_u64(
            seed ^ 0x9a7c_e2e0_f00d_51ce_u64
                ^ (step as u64).wrapping_mul(0xd1b5_4a32_d192_ed03),
        )
    }

    pub(super) fn per_identity_seed_replacement_rows(
        rollout_identities: &[usize],
        trajectory_counts: &mut [usize],
        interval: usize,
    ) -> Vec<usize> {
        let interval = interval.max(1);
        let mut rows = Vec::new();
        for (row, &identity) in rollout_identities.iter().enumerate() {
            let Some(count) = trajectory_counts.get_mut(identity) else {
                continue;
            };
            *count = count.saturating_add(1);
            if *count >= interval {
                *count %= interval;
                rows.push(row);
            }
        }
        rows
    }

    pub(super) fn e2e_seed_replacement_rows(
        rollout_identities: &[usize],
        trajectory_counts: &mut [usize],
        trajectory_interval: usize,
        step: usize,
        inject_interval: usize,
        replacements_per_interval: usize,
    ) -> Vec<usize> {
        let mut rows = per_identity_seed_replacement_rows(
            rollout_identities,
            trajectory_counts,
            trajectory_interval,
        );
        if rollout_identities.is_empty()
            || replacements_per_interval == 0
            || !step.is_multiple_of(inject_interval.max(1))
        {
            return rows;
        }

        let mut selected = vec![false; rollout_identities.len()];
        for &row in &rows {
            selected[row] = true;
        }
        let scheduled = replacements_per_interval.min(rollout_identities.len());
        let start = (step / inject_interval.max(1)).wrapping_mul(scheduled)
            % rollout_identities.len();
        let mut added = 0usize;
        for offset in 0..rollout_identities.len() {
            let row = (start + offset) % rollout_identities.len();
            if selected[row] {
                continue;
            }
            selected[row] = true;
            rows.push(row);
            added += 1;
            if added == scheduled {
                break;
            }
        }
        rows.sort_unstable();
        rows
    }

    pub(super) fn upstream_growing_repetition_reset_identities(
        rollout_identities: &[usize],
        identity_optimizer_steps: &[usize],
    ) -> Vec<usize> {
        let mut identities = rollout_identities
            .iter()
            .copied()
            .filter(|identity| {
                identity_optimizer_steps
                    .get(*identity)
                    .is_some_and(|step| *step > 0 && step.is_multiple_of(10_000))
            })
            .collect::<Vec<_>>();
        identities.sort_unstable();
        identities.dedup();
        identities
    }

    pub(super) fn seeded_values(len: usize, scale: f32, rng: &mut StdRng) -> Vec<f32> {
        let scale = scale.abs().max(f32::MIN_POSITIVE);
        (0..len)
            .map(|_| rng.random_range(-scale..scale))
            .collect::<Vec<_>>()
    }

    pub(super) fn seeded_zero_delta_output_bias(
        config: &NpaConfig,
        rank: usize,
        alpha: f32,
        seed: u64,
        output_scale: f32,
    ) -> Vec<f32> {
        let scale = output_scale.abs().max(EPSILON);
        let adapter = NpaLowRankAdapter::seeded_zero_delta(config, rank, alpha, seed);
        let mut values = adapter.to_parameter_vector();
        for value in &mut values {
            let normalized = (*value / scale).clamp(-0.95, 0.95);
            *value = normalized.atanh();
        }
        values
    }

    pub(super) fn seeded_zero_delta_chunk_output_bias(
        config: &NpaConfig,
        rank: usize,
        alpha: f32,
        seed: u64,
        chunk_size: usize,
        output_chunks: usize,
        module_layout: Option<&crate::hyper::adapter_layout::AdapterParameterLayout2d>,
    ) -> Vec<f32> {
        let adapter = NpaLowRankAdapter::seeded_zero_delta(config, rank, alpha, seed);
        let mut values = adapter.to_parameter_vector();
        if let Some(layout) = module_layout {
            return layout
                .pack(&values)
                .expect("module adapter layout matches seeded adapter");
        }
        values.resize(output_chunks.saturating_mul(chunk_size), 0.0);
        values
    }

    pub(super) fn seed_tensors(
        particle_count: usize,
        config: DirectBasisTrainConfig,
        seed_scale: f32,
        seed: u64,
        device: &BurnDevice,
    ) -> (Tensor2, Tensor2) {
        let (positions, states) =
            seed_particles_scaled(1, particle_count, 16, 2, seed, config.seed_mode, seed_scale);
        let flat_positions = positions
            .iter()
            .flat_map(|position| [position[0], position[1]])
            .collect::<Vec<_>>();
        (
            tensor(flat_positions, [particle_count, 2], device),
            tensor(states, [particle_count, 16], device),
        )
    }

    pub(super) fn write_adapters(
        examples: &mut [DirectBasisExample],
        adapters: &[BurnAdapterParams],
    ) -> AutomataResult<()> {
        for (example, adapter) in examples.iter_mut().zip(adapters) {
            example.adapter = adapter.to_adapter()?;
        }
        Ok(())
    }

    pub(super) fn pool_age_stratum(
        age: usize,
        max_age_steps: usize,
        strata: usize,
    ) -> usize {
        if strata < 2 || max_age_steps == 0 {
            return 0;
        }
        age.min(max_age_steps.saturating_sub(1))
            .saturating_mul(strata)
            .checked_div(max_age_steps)
            .unwrap_or(0)
            .min(strata - 1)
    }

    pub(super) fn sample_pool_indices_by_age(
        rng: &mut StdRng,
        ages: &[usize],
        batch_size: usize,
        fresh_seed_rows: usize,
        max_age_steps: Option<usize>,
        age_strata: usize,
    ) -> Vec<usize> {
        sample_pool_indices_by_age_with_event_preference(
            rng,
            ages,
            batch_size,
            fresh_seed_rows,
            max_age_steps,
            age_strata,
            None,
        )
    }

    pub(super) fn sample_pool_indices_by_age_with_event_preference(
        rng: &mut StdRng,
        ages: &[usize],
        batch_size: usize,
        fresh_seed_rows: usize,
        max_age_steps: Option<usize>,
        age_strata: usize,
        event_preference: Option<BurnPoolEventPreference>,
    ) -> Vec<usize> {
        let sample_count = batch_size.min(ages.len());
        if sample_count == 0 {
            return Vec::new();
        }

        let fresh_count = fresh_seed_rows.min(sample_count);
        let mut available = (0..ages.len()).collect::<Vec<_>>();
        available.shuffle(rng);
        let mut selected = available.drain(..fresh_count).collect::<Vec<_>>();
        if let Some(preference) = event_preference {
            let fresh_event_rows = usize::from(pool_age_crosses_preferred_event(0, preference))
                * fresh_count;
            let preferred_rows = preference
                .min_rows
                .min(sample_count)
                .saturating_sub(fresh_event_rows)
                .min(sample_count - selected.len());
            if preferred_rows > 0 {
                let mut candidates = available
                    .iter()
                    .copied()
                    .filter(|index| {
                        pool_age_crosses_preferred_event(ages[*index], preference)
                    })
                    .collect::<Vec<_>>();
                candidates.shuffle(rng);
                candidates.truncate(preferred_rows);
                available.retain(|index| !candidates.contains(index));
                selected.extend(candidates);
            }
        }

        let persistent_count = sample_count - selected.len();
        let Some(max_age_steps) =
            max_age_steps.filter(|max_age| *max_age > 0 && age_strata >= 2)
        else {
            available.shuffle(rng);
            selected.extend(available.into_iter().take(persistent_count));
            return selected;
        };
        let mut buckets = vec![Vec::new(); age_strata];
        for index in available {
            buckets[pool_age_stratum(ages[index], max_age_steps, age_strata)].push(index);
        }
        for bucket in &mut buckets {
            bucket.shuffle(rng);
        }

        let mut missing = 0usize;
        for slot in 0..persistent_count {
            let stratum = slot % age_strata;
            if let Some(index) = buckets[stratum].pop() {
                selected.push(index);
            } else {
                missing += 1;
            }
        }
        if missing > 0 {
            let mut remaining = buckets.into_iter().flatten().collect::<Vec<_>>();
            remaining.shuffle(rng);
            selected.extend(remaining.into_iter().take(missing));
        }
        selected
    }

    pub(super) fn pool_age_crosses_preferred_event(
        age: usize,
        preference: BurnPoolEventPreference,
    ) -> bool {
        if preference.interval_steps == 0 || preference.lookahead_steps == 0 {
            return false;
        }
        let after = age.saturating_add(preference.lookahead_steps);
        let lower = age
            .saturating_add(1)
            .max(preference.start_step)
            .max(1);
        let next = lower.div_ceil(preference.interval_steps) * preference.interval_steps;
        next <= after && (preference.end_step == 0 || next <= preference.end_step)
    }

    impl BurnDeviceParticlePool {
        const MAX_RECOVERABLE_POSITION: f32 = 4.0;
        const MAX_RECOVERABLE_STATE: f32 = 32.0;

        pub(super) fn new(
            pool_size: usize,
            particle_count: usize,
            state_dims: usize,
            seed_scale: f32,
            config: DirectBasisTrainConfig,
            device: &BurnDevice,
        ) -> Self {
            let (positions, states) = seed_particles_scaled(
                pool_size,
                particle_count,
                state_dims,
                2,
                config.seed,
                config.seed_mode,
                seed_scale,
            );
            let position_values = positions
                .iter()
                .flat_map(|position| [position[0], position[1]])
                .collect::<Vec<_>>();
            Self::from_flat_values(
                position_values,
                states,
                pool_size,
                particle_count,
                state_dims,
                device,
            )
            .expect("generated particle pool has canonical tensor shapes")
        }

        pub(super) fn from_flat_values(
            positions: Vec<f32>,
            states: Vec<f32>,
            pool_size: usize,
            particle_count: usize,
            state_dims: usize,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            if pool_size == 0
                || particle_count == 0
                || state_dims == 0
                || positions.len() != pool_size * particle_count * 2
                || states.len() != pool_size * particle_count * state_dims
                || positions
                    .iter()
                    .chain(&states)
                    .any(|value| !value.is_finite())
            {
                return Err(AutomataError::InvalidArgument(
                    "device particle pool values have invalid shapes or non-finite values"
                        .to_owned(),
                ));
            }
            let inner_device = Device::<InnerBackend>::from(device.clone());
            let positions = Tensor::<InnerBackend, 3>::from_data(
                TensorData::new(positions, [pool_size, particle_count, 2]),
                &inner_device,
            );
            let states = Tensor::<InnerBackend, 3>::from_data(
                TensorData::new(states, [pool_size, particle_count, state_dims]),
                &inner_device,
            );
            Ok(Self {
                initial_positions: positions.clone(),
                initial_states: states.clone(),
                positions,
                states,
                ages: vec![0; pool_size],
                pool_size,
                particle_count,
                state_dims,
            })
        }

        pub(super) fn reset(&mut self) {
            self.positions = self.initial_positions.clone();
            self.states = self.initial_states.clone();
            self.ages.fill(0);
        }

        pub(super) fn target2d_snapshot(
            &self,
        ) -> AutomataResult<Target2dParticlePoolSnapshot> {
            Ok(Target2dParticlePoolSnapshot {
                positions: tensor3_snapshot(
                    "target2d.pool.positions",
                    self.positions.clone(),
                )?,
                states: tensor3_snapshot("target2d.pool.states", self.states.clone())?,
                ages: self.ages.clone(),
            })
        }

        pub(super) fn restore_target2d_snapshot(
            &mut self,
            snapshot: &Target2dParticlePoolSnapshot,
            device: &BurnDevice,
        ) -> AutomataResult<()> {
            let expected_positions = [self.pool_size, self.particle_count, 2];
            let expected_states = [self.pool_size, self.particle_count, self.state_dims];
            if snapshot.positions.shape != expected_positions
                || snapshot.states.shape != expected_states
            {
                return Err(AutomataError::InvalidArgument(format!(
                    "Target2D particle-pool checkpoint shapes {:?}/{:?} != expected {:?}/{:?}",
                    snapshot.positions.shape,
                    snapshot.states.shape,
                    expected_positions,
                    expected_states,
                )));
            }
            self.positions = tensor3_from_snapshot(&snapshot.positions, device)?;
            self.states = tensor3_from_snapshot(&snapshot.states, device)?;
            if snapshot.ages.is_empty() {
                self.ages.fill(0);
            } else if snapshot.ages.len() == self.pool_size {
                self.ages.clone_from(&snapshot.ages);
            } else {
                return Err(AutomataError::InvalidArgument(format!(
                    "Target2D particle-pool checkpoint age count {} != expected {}",
                    snapshot.ages.len(),
                    self.pool_size,
                )));
            }
            Ok(())
        }

        pub(super) fn restore_unhealthy_batch(
            &self,
            pool_indices: &[usize],
            x: Tensor3,
            s: Tensor3,
        ) -> AutomataResult<(Tensor3, Tensor3, Tensor1Bool)> {
            let x_dims = x.shape().dims::<3>();
            let s_dims = s.shape().dims::<3>();
            if x_dims != [pool_indices.len(), self.particle_count, 2]
                || s_dims != [pool_indices.len(), self.particle_count, self.state_dims]
            {
                return Err(AutomataError::InvalidArgument(
                    "device particle-pool recovery received incompatible batch shapes".to_owned(),
                ));
            }
            if pool_indices.is_empty() {
                let device = x.device();
                return Ok((
                    x,
                    s,
                    Tensor::<BurnBackend, 1, Bool>::empty([0], &device),
                ));
            }

            let x_healthy = x
                .clone()
                .is_finite()
                .all_dim(2)
                .all_dim(1)
                .bool_and(
                    x.clone()
                        .abs()
                        .lower_equal_elem(Self::MAX_RECOVERABLE_POSITION)
                        .all_dim(2)
                        .all_dim(1),
                );
            let s_healthy = s
                .clone()
                .is_finite()
                .all_dim(2)
                .all_dim(1)
                .bool_and(
                    s.clone()
                        .abs()
                        .lower_equal_elem(Self::MAX_RECOVERABLE_STATE)
                        .all_dim(2)
                        .all_dim(1),
                );
            let unhealthy = x_healthy.bool_and(s_healthy).bool_not();
            let unhealthy_rows = unhealthy
                .clone()
                .squeeze_dim::<2>(2)
                .squeeze_dim::<1>(1);
            let inner_device = self.positions.device();
            let indices = inner_index_tensor(pool_indices, &inner_device);
            let initial_x = Tensor::<BurnBackend, 3>::from_inner(
                self.initial_positions.clone().select(0, indices.clone()),
            );
            let initial_s = Tensor::<BurnBackend, 3>::from_inner(
                self.initial_states.clone().select(0, indices),
            );
            Ok((
                x.mask_where(unhealthy.clone().expand(x_dims), initial_x),
                s.mask_where(unhealthy.expand(s_dims), initial_s),
                unhealthy_rows,
            ))
        }

        pub(super) fn sample_batch(
            &self,
            rng: &mut StdRng,
            batch_size: usize,
            replace_seed: bool,
            _seed_scale: f32,
            config: DirectBasisTrainConfig,
            device: &BurnDevice,
        ) -> AutomataResult<BurnPoolBatch> {
            self.sample_batch_with_fresh_rows(
                rng,
                batch_size,
                BurnPoolSampling {
                    fresh_seed_rows: usize::from(replace_seed),
                    max_age_steps: None,
                    age_strata: 0,
                    event_preference: None,
                },
                config,
                device,
            )
        }

        pub(super) fn sample_batch_with_fresh_rows(
            &self,
            rng: &mut StdRng,
            batch_size: usize,
            sampling: BurnPoolSampling,
            config: DirectBasisTrainConfig,
            device: &BurnDevice,
        ) -> AutomataResult<BurnPoolBatch> {
            let pool_indices = sample_pool_indices_by_age_with_event_preference(
                rng,
                &self.ages,
                batch_size,
                sampling.fresh_seed_rows,
                sampling.max_age_steps,
                sampling.age_strata,
                sampling.event_preference,
            );
            let inner_device = self.positions.device();
            let indices = inner_index_tensor(&pool_indices, &inner_device);
            let mut x = Tensor::<BurnBackend, 3>::from_inner(
                self.positions.clone().select(0, indices.clone()),
            );
            let mut s =
                Tensor::<BurnBackend, 3>::from_inner(self.states.clone().select(0, indices));
            let mut ages = pool_indices
                .iter()
                .map(|index| self.ages[*index])
                .collect::<Vec<_>>();

            let fresh_seed_rows = sampling.fresh_seed_rows.min(pool_indices.len());
            let mut fresh_batch_rows = ages
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(batch_row, age)| {
                    (batch_row < fresh_seed_rows
                        || sampling
                            .max_age_steps
                            .is_some_and(|max_age| age >= max_age))
                    .then_some(batch_row)
                })
                .collect::<Vec<_>>();
            if let Some(preference) = sampling.event_preference
                && pool_age_crosses_preferred_event(0, preference)
            {
                let preferred_rows = ages
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(batch_row, age)| {
                        let effective_age = if fresh_batch_rows.contains(batch_row) {
                            0
                        } else {
                            *age
                        };
                        pool_age_crosses_preferred_event(effective_age, preference)
                    })
                    .count();
                let mut deficit = preference
                    .min_rows
                    .min(pool_indices.len())
                    .saturating_sub(preferred_rows);
                if deficit > 0 {
                    for (batch_row, age) in ages.iter().copied().enumerate() {
                        if deficit == 0 {
                            break;
                        }
                        if !fresh_batch_rows.contains(&batch_row)
                            && !pool_age_crosses_preferred_event(age, preference)
                        {
                            fresh_batch_rows.push(batch_row);
                            deficit -= 1;
                        }
                    }
                }
            }
            fresh_batch_rows.sort_unstable();
            if !fresh_batch_rows.is_empty() {
                for batch_row in fresh_batch_rows.iter().copied() {
                    ages[batch_row] = 0;
                }
                let initial_rows = (0..fresh_batch_rows.len())
                    .map(|_| rng.random_range(0..self.pool_size))
                    .collect::<Vec<_>>();
                let inner_initial_index = inner_index_tensor(&initial_rows, &inner_device);
                let replacement_rows = fresh_batch_rows
                    .iter()
                    .map(|row| *row as i64)
                    .collect::<Vec<_>>();
                let replacement = Tensor::<BurnBackend, 1, Int>::from_data(
                    TensorData::new(replacement_rows, [fresh_batch_rows.len()]),
                    device,
                );
                let new_positions = Tensor::<BurnBackend, 3>::from_inner(
                    self.initial_positions
                        .clone()
                        .select(0, inner_initial_index.clone()),
                );
                let position_delta = new_positions - x.clone().select(0, replacement.clone());
                x = x.select_assign(
                    0,
                    replacement.clone(),
                    position_delta,
                    IndexingUpdateOp::Add,
                );
                let new_states = Tensor::<BurnBackend, 3>::from_inner(
                    self.initial_states
                        .clone()
                        .select(0, inner_initial_index),
                );
                let state_delta = new_states - s.clone().select(0, replacement.clone());
                s = s.select_assign(0, replacement, state_delta, IndexingUpdateOp::Add);
            }

            if config.brush_size > 0.0 && !pool_indices.is_empty() {
                let center_indices = (0..pool_indices.len())
                    .map(|batch| {
                        (batch * self.particle_count + rng.random_range(0..self.particle_count))
                            as i64
                    })
                    .collect::<Vec<_>>();
                let center_indices = Tensor::<BurnBackend, 1, Int>::from_data(
                    TensorData::new(center_indices, [pool_indices.len()]),
                    device,
                );
                let centers = x
                    .clone()
                    .reshape([pool_indices.len() * self.particle_count, 2])
                    .select(0, center_indices)
                    .reshape([pool_indices.len(), 1, 2])
                    .expand([pool_indices.len(), self.particle_count, 2]);
                let diff = x.clone() - centers;
                let damaged = diff
                    .clone()
                    .mul(diff)
                    .sum_dim(2)
                    .lower_elem(config.brush_size * config.brush_size)
                    .expand([pool_indices.len(), self.particle_count, self.state_dims]);
                s = s.mask_fill(damaged, 0.0);
            }

            Ok(BurnPoolBatch {
                pool_indices,
                ages,
                x,
                s,
            })
        }

        pub(super) fn update_batch(
            &mut self,
            pool_indices: &[usize],
            x: Tensor3,
            s: Tensor3,
        ) -> AutomataResult<()> {
            if pool_indices.is_empty() {
                return Ok(());
            }
            let inner_device = self.positions.device();
            let indices = inner_index_tensor(pool_indices, &inner_device);
            let position_delta = x.inner() - self.positions.clone().select(0, indices.clone());
            self.positions = self.positions.clone().select_assign(
                0,
                indices.clone(),
                position_delta,
                IndexingUpdateOp::Add,
            );
            let state_delta = s.inner() - self.states.clone().select(0, indices.clone());
            self.states = self.states.clone().select_assign(
                0,
                indices,
                state_delta,
                IndexingUpdateOp::Add,
            );
            Ok(())
        }

        pub(super) fn update_batch_with_ages(
            &mut self,
            pool_indices: &[usize],
            ages: &[usize],
            x: Tensor3,
            s: Tensor3,
        ) -> AutomataResult<()> {
            if pool_indices.len() != ages.len() {
                return Err(AutomataError::InvalidArgument(format!(
                    "device particle-pool update has {} indices but {} ages",
                    pool_indices.len(),
                    ages.len(),
                )));
            }
            self.update_batch(pool_indices, x, s)?;
            for (index, age) in pool_indices.iter().copied().zip(ages.iter().copied()) {
                self.ages[index] = age;
            }
            Ok(())
        }
    }

    impl BurnE2eDeviceParticlePool {
        pub(super) fn new(
            capacity: usize,
            particle_count: usize,
            state_dims: usize,
            slots_per_example: usize,
            device: &BurnDevice,
        ) -> Self {
            let inner_device = Device::<InnerBackend>::from(device.clone());
            Self {
                positions: Tensor::<InnerBackend, 3>::zeros(
                    [capacity, particle_count, 2],
                    &inner_device,
                ),
                states: Tensor::<InnerBackend, 3>::zeros(
                    [capacity, particle_count, state_dims],
                    &inner_device,
                ),
                slot_examples: vec![None; capacity],
                example_slots: HashMap::with_capacity(capacity),
                next_evict: 0,
                capacity,
                particle_count,
                state_dims,
                slots_per_example: slots_per_example.max(1),
            }
        }

        pub(super) fn sample_batch(
            &mut self,
            example_indices: &[usize],
            rng: &mut StdRng,
            seed_replacement_rows: &[usize],
            seed_scale: f32,
            config: DirectBasisTrainConfig,
            device: &BurnDevice,
        ) -> AutomataResult<BurnE2ePoolBatch> {
            if example_indices.len() > self.capacity {
                return Err(AutomataError::InvalidArgument(format!(
                    "device particle pool capacity {} is smaller than batch {}",
                    self.capacity,
                    example_indices.len()
                )));
            }
            let mut slots = Vec::with_capacity(example_indices.len());
            let mut new_slots = Vec::new();
            let mut replica_choices = HashMap::<usize, Vec<usize>>::new();
            for &example in example_indices {
                let choices = replica_choices.entry(example).or_insert_with(|| {
                    let mut choices = (0..self.slots_per_example).collect::<Vec<_>>();
                    choices.shuffle(rng);
                    choices
                });
                let replica = choices.pop().ok_or_else(|| {
                    AutomataError::InvalidArgument(format!(
                        "requested more than {} rollout replicas for example {example}",
                        self.slots_per_example
                    ))
                })?;
                let key = (example, replica);
                if let Some(&slot) = self.example_slots.get(&key) {
                    slots.push(slot);
                    continue;
                }
                let slot = self.allocate_slot(&slots);
                if let Some(previous) = self.slot_examples[slot].replace(key) {
                    self.example_slots.remove(&previous);
                }
                self.example_slots.insert(key, slot);
                slots.push(slot);
                new_slots.push(slot);
            }
            if !new_slots.is_empty() {
                let seed = config.seed ^ rng.random::<u64>();
                let (positions, states) = seed_particles_scaled(
                    new_slots.len(),
                    self.particle_count,
                    self.state_dims,
                    2,
                    seed,
                    config.seed_mode,
                    seed_scale,
                );
                let position_values = positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]])
                    .collect::<Vec<_>>();
                let inner_device = self.positions.device();
                let indices = inner_index_tensor(&new_slots, &inner_device);
                let new_positions = Tensor::<InnerBackend, 3>::from_data(
                        TensorData::new(
                            position_values,
                            [new_slots.len(), self.particle_count, 2],
                        ),
                        &inner_device,
                    );
                let position_delta = new_positions - self.positions.clone().select(0, indices.clone());
                self.positions = self.positions.clone().select_assign(
                    0,
                    indices.clone(),
                    position_delta,
                    IndexingUpdateOp::Add,
                );
                let new_states = Tensor::<InnerBackend, 3>::from_data(
                        TensorData::new(
                            states,
                            [new_slots.len(), self.particle_count, self.state_dims],
                        ),
                        &inner_device,
                    );
                let state_delta = new_states - self.states.clone().select(0, indices.clone());
                self.states = self.states.clone().select_assign(
                    0,
                    indices,
                    state_delta,
                    IndexingUpdateOp::Add,
                );
            }

            let inner_device = self.positions.device();
            let slot_indices = inner_index_tensor(&slots, &inner_device);
            let mut x = Tensor::<BurnBackend, 3>::from_inner(
                self.positions.clone().select(0, slot_indices.clone()),
            );
            let mut s = Tensor::<BurnBackend, 3>::from_inner(
                self.states.clone().select(0, slot_indices),
            );
            let mut replacement_rows = seed_replacement_rows
                .iter()
                .copied()
                .filter(|row| *row < slots.len())
                .collect::<Vec<_>>();
            replacement_rows.sort_unstable();
            replacement_rows.dedup();
            let seed_replacements = replacement_rows.len();
            if seed_replacements > 0 {
                let seed = config.seed ^ rng.random::<u64>();
                let (positions, states) = seed_particles_scaled(
                    seed_replacements,
                    self.particle_count,
                    self.state_dims,
                    2,
                    seed,
                    config.seed_mode,
                    seed_scale,
                );
                let position_values = positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]])
                    .collect::<Vec<_>>();
                let replacement = Tensor::<BurnBackend, 1, Int>::from_data(
                    TensorData::new(
                        replacement_rows
                            .iter()
                            .map(|row| *row as i64)
                            .collect::<Vec<_>>(),
                        [seed_replacements],
                    ),
                    device,
                );
                let new_positions = tensor3(
                    position_values,
                    [seed_replacements, self.particle_count, 2],
                    device,
                );
                let position_delta = new_positions - x.clone().select(0, replacement.clone());
                x = x.select_assign(
                    0,
                    replacement.clone(),
                    position_delta,
                    IndexingUpdateOp::Add,
                );
                let new_states = tensor3(
                    states,
                    [seed_replacements, self.particle_count, self.state_dims],
                    device,
                );
                let state_delta = new_states - s.clone().select(0, replacement.clone());
                s = s.select_assign(
                    0,
                    replacement,
                    state_delta,
                    IndexingUpdateOp::Add,
                );
            }
            if config.brush_size > 0.0 && !slots.is_empty() {
                let center_particles = (0..slots.len())
                    .map(|_| rng.random_range(0..self.particle_count))
                    .collect::<Vec<_>>();
                let centers = gather_live_particle_centers(x.clone(), &center_particles, device)
                    .expand([slots.len(), self.particle_count, 2]);
                let diff = x.clone() - centers;
                let damaged = diff
                    .clone()
                    .mul(diff)
                    .sum_dim(2)
                    .lower_elem(config.brush_size * config.brush_size)
                    .expand([slots.len(), self.particle_count, self.state_dims]);
                s = s.mask_fill(damaged, 0.0);
            }
            Ok(BurnE2ePoolBatch {
                slots,
                x,
                s,
                seed_replacements,
            })
        }

        pub(super) fn reset_examples(&mut self, examples: &[usize]) {
            for &example in examples {
                for replica in 0..self.slots_per_example {
                    let key = (example, replica);
                    if let Some(slot) = self.example_slots.remove(&key) {
                        self.slot_examples[slot] = None;
                    }
                }
            }
        }

        pub(super) fn update_batch(
            &mut self,
            slots: &[usize],
            x: Tensor3,
            s: Tensor3,
        ) -> AutomataResult<()> {
            if slots.is_empty() {
                return Ok(());
            }
            let inner_device = self.positions.device();
            let indices = inner_index_tensor(slots, &inner_device);
            let persisted_x = x.inner();
            let persisted_x = persisted_x
                .clone()
                .mask_fill(persisted_x.is_finite().bool_not(), 0.0)
                .clamp(-1.0, 1.0);
            let position_delta =
                persisted_x - self.positions.clone().select(0, indices.clone());
            self.positions = self.positions.clone().select_assign(
                0,
                indices.clone(),
                position_delta,
                IndexingUpdateOp::Add,
            );
            // Upstream persists mature recurrent state without amplitude clipping. Keep a
            // generous finite safety bound, but do not erase valid attractor state at +/-1.
            let persisted_s = s.inner();
            let persisted_s = persisted_s
                .clone()
                .mask_fill(persisted_s.is_finite().bool_not(), 0.0)
                .clamp(-32.0, 32.0);
            let state_delta = persisted_s - self.states.clone().select(0, indices.clone());
            self.states = self.states.clone().select_assign(
                0,
                indices,
                state_delta,
                IndexingUpdateOp::Add,
            );
            Ok(())
        }

        pub(super) fn snapshot(&self) -> AutomataResult<E2eParticlePoolSnapshot> {
            Ok(E2eParticlePoolSnapshot {
                positions: tensor3_snapshot("pool.positions", self.positions.clone())?,
                states: tensor3_snapshot("pool.states", self.states.clone())?,
                slot_examples: self.slot_examples.clone(),
                next_evict: self.next_evict,
                slots_per_example: self.slots_per_example,
            })
        }

        pub(super) fn restore(
            snapshot: &E2eParticlePoolSnapshot,
            config: BurnE2eRolloutTrainConfig,
            device: &BurnDevice,
        ) -> AutomataResult<Self> {
            let position_shape: [usize; 3] = snapshot
                .positions
                .shape
                .clone()
                .try_into()
                .map_err(|_| {
                    AutomataError::InvalidArgument(
                        "checkpoint pool positions are not rank three".to_string(),
                    )
                })?;
            let state_shape: [usize; 3] = snapshot.states.shape.clone().try_into().map_err(|_| {
                AutomataError::InvalidArgument(
                    "checkpoint pool states are not rank three".to_string(),
                )
            })?;
            if position_shape[0] != state_shape[0]
                || position_shape[1] != state_shape[1]
                || position_shape[2] != 2
                || state_shape[2] != 16
                || position_shape[0] != snapshot.slot_examples.len()
                || position_shape[0] != config.pool_capacity
                || position_shape[1] != config.rollout_particles
                || snapshot.slots_per_example != config.pool_slots_per_example
            {
                return Err(AutomataError::InvalidArgument(format!(
                    "checkpoint particle pool shape {:?}/{:?} is incompatible with capacity={} particles={} slots_per_example={}",
                    position_shape,
                    state_shape,
                    config.pool_capacity,
                    config.rollout_particles,
                    config.pool_slots_per_example,
                )));
            }
            let mut example_slots = HashMap::with_capacity(snapshot.slot_examples.len());
            for (slot, key) in snapshot.slot_examples.iter().enumerate() {
                if let Some(key) = key {
                    example_slots.insert(*key, slot);
                }
            }
            Ok(Self {
                positions: tensor3_from_snapshot(&snapshot.positions, device)?,
                states: tensor3_from_snapshot(&snapshot.states, device)?,
                slot_examples: snapshot.slot_examples.clone(),
                example_slots,
                next_evict: snapshot.next_evict % config.pool_capacity.max(1),
                capacity: config.pool_capacity,
                particle_count: config.rollout_particles,
                state_dims: 16,
                slots_per_example: config.pool_slots_per_example,
            })
        }

        pub(super) fn allocate_slot(&mut self, protected: &[usize]) -> usize {
            if let Some(slot) = self.slot_examples.iter().position(Option::is_none) {
                return slot;
            }
            for _ in 0..self.capacity {
                let slot = self.next_evict;
                self.next_evict = (self.next_evict + 1) % self.capacity;
                if !protected.contains(&slot) {
                    return slot;
                }
            }
            unreachable!("pool capacity is validated against batch size")
        }
    }

    pub(super) fn gather_live_particle_centers(
        positions: Tensor3,
        particle_indices: &[usize],
        device: &BurnDevice,
    ) -> Tensor3 {
        let indices = particle_indices
            .iter()
            .flat_map(|index| [*index as i64, *index as i64])
            .collect::<Vec<_>>();
        let indices = Tensor::<BurnBackend, 3, Int>::from_data(
            TensorData::new(indices, [particle_indices.len(), 1, 2]),
            device,
        );
        positions.gather(1, indices)
    }

    pub(super) fn inner_index_tensor(indices: &[usize], device: &Device<InnerBackend>) -> Tensor1IntInner {
        Tensor::from_data(
            TensorData::new(
                indices.iter().map(|index| *index as i64).collect::<Vec<_>>(),
                [indices.len()],
            ),
            device,
        )
    }

    pub(super) struct PhaseBatchSampler {
        len: usize,
        batch_size: usize,
        order: Vec<usize>,
        cursor: usize,
    }

    impl PhaseBatchSampler {
        pub(super) fn new(len: usize, requested: usize, rng: &mut StdRng) -> Self {
            let batch_size = if requested == 0 {
                len
            } else {
                requested.min(len)
            };
            let mut order = (0..len).collect::<Vec<_>>();
            order.shuffle(rng);
            Self {
                len,
                batch_size,
                order,
                cursor: 0,
            }
        }

        pub(super) fn next_batch(&mut self, rng: &mut StdRng) -> Vec<usize> {
            if self.len == 0 || self.batch_size == 0 {
                return Vec::new();
            }
            if self.batch_size >= self.len {
                let mut indices = (0..self.len).collect::<Vec<_>>();
                indices.shuffle(rng);
                return indices;
            }

            let mut indices = Vec::with_capacity(self.batch_size);
            while indices.len() < self.batch_size {
                if self.cursor >= self.order.len() {
                    self.reshuffle_excluding(rng, &indices);
                }
                let idx = self.order[self.cursor];
                self.cursor += 1;
                if !indices.contains(&idx) {
                    indices.push(idx);
                }
            }
            indices
        }

        pub(super) fn reshuffle_excluding(&mut self, rng: &mut StdRng, exclude: &[usize]) {
            self.order = (0..self.len)
                .filter(|idx| !exclude.contains(idx))
                .collect::<Vec<_>>();
            self.order.shuffle(rng);
            self.cursor = 0;
        }
    }
