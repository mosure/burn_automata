//! Target preparation, condition loading, sampling, and device particle pools.

use super::*;

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
                .saturating_add(example.target.point_count().saturating_mul(2));
            bytes.saturating_add(floats.saturating_mul(std::mem::size_of::<f32>()))
        })
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

    pub(super) fn spawn_e2e_cpu_batch_prefetch(
        examples: &[BurnE2eRolloutExample],
        conditions: &BurnE2eConditionCache,
        indices: Vec<usize>,
        config: BurnE2eRolloutTrainConfig,
        targets_cached: bool,
    ) -> AutomataResult<BurnE2eCpuBatchPrefetch> {
        let mut target_inputs = Vec::with_capacity(indices.len());
        if !targets_cached {
            for &idx in &indices {
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
        }
        let condition_paths = conditions.dynamic_dino_paths_for_indices(&indices)?;
        let pending_indices = indices.clone();
        Ok(BurnE2eCpuBatchPrefetch {
            indices: pending_indices,
            handle: thread::spawn(move || {
                prepare_e2e_cpu_batch(indices, target_inputs, condition_paths, config)
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
        target_inputs: Vec<BurnE2eCpuTargetInput>,
        condition_paths: Option<Vec<PathBuf>>,
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
                Some(paths) => prepare_dino_condition_batch_for_prefetch(paths, config.dino_image_size)
                    .map(Some),
                None => Ok(None),
            },
        );
        let targets = targets?;
        let prepared_dino = prepared_dino?;
        Ok(BurnE2ePreparedCpuBatch {
            indices,
            targets,
            prepared_dino,
        })
    }

    pub(super) fn prepare_e2e_cpu_target(
        input: BurnE2eCpuTargetInput,
        config: DirectBasisTrainConfig,
    ) -> AutomataResult<BurnE2ePreparedTargetExample> {
        let pixels = config.loss_config.image_size * config.loss_config.image_size;
        let render = render_target_2d_splat(&input.target, config.loss_config)?;
        let foreground = target_2d_foreground_mask(&input.target, config.loss_config)?;
        let target_foreground_scale = pixels as f32 / foreground.iter().sum::<f32>().max(1.0);
        let target_mean = input.target.mean_position();
        let target_positions = input
            .target
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
            pixel_size: input.target.pixel_size,
            target_points: input.target.point_count(),
            particle_count: input.particle_count.max(1),
            update_prob: input.update_prob,
            seed_scale: input.seed_scale,
            target_cpu: input.target,
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
        image_size: usize,
    ) -> Result<DinoVitsPreparedConditionBatch, String> {
        let images = paths
            .into_par_iter()
            .map(|path| load_dino_condition_image(&path).map_err(|err| err.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        DinoVitsPreparedConditionBatch::from_conditions(&images, image_size)
            .map_err(|err| err.to_string())
    }

    #[cfg(not(feature = "dino"))]
    pub(super) fn prepare_dino_condition_batch_for_prefetch(
        _paths: Vec<PathBuf>,
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
            config.example_batch_size.max(1)
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
            let indices = (0..self.examples).collect::<Vec<_>>();
            let values = tensor3_vec(self.select(&indices)?.inner())?;
            let row_len = self.token_count * self.embed_dims;
            let mut sum = 0.0_f64;
            let mut pairs = 0usize;
            for lhs in 0..self.examples {
                for rhs in lhs + 1..self.examples {
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
            let Some(teachers) = &self.teacher_vectors else {
                return Ok(None);
            };
            if self.examples < 2 {
                return Ok(None);
            }
            let dims = teachers.shape().dims::<2>();
            let values = tensor_vec(teachers.clone().inner())?;
            let mut sum = 0.0_f64;
            let mut pairs = 0usize;
            for lhs in 0..self.examples {
                for rhs in lhs + 1..self.examples {
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
                return source.encode_preprocessed(prepared, indices.len(), self.token_count, self.embed_dims);
            }
            self.select(indices)
        }

        pub(super) fn dynamic_dino_paths_for_indices(
            &self,
            indices: &[usize],
        ) -> AutomataResult<Option<Vec<PathBuf>>> {
            if indices.iter().any(|idx| *idx >= self.examples) {
                return Err(AutomataError::InvalidArgument(
                    "HyperNPA e2e condition cache index out of bounds".to_string(),
                ));
            }
            #[cfg(feature = "dino")]
            if let BurnE2eConditionValues::DynamicDino(source) = &self.values {
                return indices
                    .iter()
                    .map(|idx| {
                        source.paths.get(*idx).cloned().ok_or_else(|| {
                            AutomataError::InvalidArgument(
                                "DINO condition source index out of bounds".to_string(),
                            )
                        })
                    })
                    .collect::<AutomataResult<Vec<_>>>()
                    .map(Some);
            }
            Ok(None)
        }

        pub(super) fn feature_bytes(&self) -> usize {
            #[cfg(feature = "dino")]
            if matches!(self.values, BurnE2eConditionValues::DynamicDino(_)) {
                return 0;
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
                BurnE2eConditionValues::DynamicDino(_) => "dino-on-demand-device",
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
            let mut chunks = Vec::with_capacity(indices.len().div_ceil(self.batch_size));
            for chunk_indices in indices.chunks(self.batch_size) {
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
            if dims != [indices.len(), token_count, embed_dims] {
                return Err(AutomataError::InvalidArgument(format!(
                    "on-demand DINO condition tensor shape {:?} != [{}, {}, {}]",
                    dims,
                    indices.len(),
                    token_count,
                    embed_dims
                )));
            }
            Ok(Tensor::<BurnBackend, 3>::from_inner(encoded))
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

    impl BurnDeviceParticlePool {
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
            let inner_device = Device::<InnerBackend>::from(device.clone());
            Self {
                positions: Tensor::<InnerBackend, 3>::from_data(
                    TensorData::new(position_values, [pool_size, particle_count, 2]),
                    &inner_device,
                ),
                states: Tensor::<InnerBackend, 3>::from_data(
                    TensorData::new(states, [pool_size, particle_count, state_dims]),
                    &inner_device,
                ),
                pool_size,
                particle_count,
                state_dims,
            }
        }

        pub(super) fn sample_batch(
            &self,
            rng: &mut StdRng,
            batch_size: usize,
            replace_seed: bool,
            seed_scale: f32,
            config: DirectBasisTrainConfig,
            device: &BurnDevice,
        ) -> AutomataResult<BurnPoolBatch> {
            let mut pool_indices = (0..self.pool_size).collect::<Vec<_>>();
            pool_indices.shuffle(rng);
            pool_indices.truncate(batch_size.min(self.pool_size));
            let inner_device = self.positions.device();
            let indices = inner_index_tensor(&pool_indices, &inner_device);
            let mut x = Tensor::<BurnBackend, 3>::from_inner(
                self.positions.clone().select(0, indices.clone()),
            );
            let mut s =
                Tensor::<BurnBackend, 3>::from_inner(self.states.clone().select(0, indices));

            if replace_seed && !pool_indices.is_empty() {
                let seed = config.seed ^ rng.random::<u64>();
                let (seed_positions, seed_states) = seed_particles_scaled(
                    1,
                    self.particle_count,
                    self.state_dims,
                    2,
                    seed,
                    config.seed_mode,
                    seed_scale,
                );
                let position_values = seed_positions
                    .iter()
                    .flat_map(|position| [position[0], position[1]])
                    .collect::<Vec<_>>();
                let replacement = Tensor::<BurnBackend, 1, Int>::from_data(
                    TensorData::new(vec![0_i64], [1]),
                    device,
                );
                let new_positions = tensor3(
                    position_values,
                    [1, self.particle_count, 2],
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
                    seed_states,
                    [1, self.particle_count, self.state_dims],
                    device,
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
            let persisted_x = x.inner().clamp(-1.0, 1.0);
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
            let persisted_s = s.inner().clamp(-32.0, 32.0);
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
