//! Dense and CubeCL perception forwards with their Burn autodiff boundary.

use super::*;

    pub(super) fn rollout_dense_perception(
        x: &Tensor2,
        s: &Tensor2,
        config: DirectBasisTrainConfig,
    ) -> Tensor2 {
        let feature_x = if config.stopgrad_pos {
            detach2(x.clone())
        } else {
            x.clone()
        };
        let feature_s = if config.stopgrad_state {
            detach2(s.clone())
        } else {
            s.clone()
        };
        match perception_backend_effective(config) {
            PerceptionRolloutBackend::Dense => dense_perception(&feature_x, &feature_s, config),
            PerceptionRolloutBackend::TiledAdjoint => perception_tiled_adjoint_batch(
                feature_x.unsqueeze_dim::<3>(0),
                feature_s.unsqueeze_dim::<3>(0),
                config,
            )
            .squeeze_dim::<2>(0),
            PerceptionRolloutBackend::Auto => unreachable!("auto perception backend resolved"),
        }
    }

    pub(super) fn rollout_dense_perception_batch(
        x: &Tensor3,
        s: &Tensor3,
        config: DirectBasisTrainConfig,
    ) -> Tensor3 {
        let feature_x = if config.stopgrad_pos {
            detach3(x.clone())
        } else {
            x.clone()
        };
        let feature_s = if config.stopgrad_state {
            detach3(s.clone())
        } else {
            s.clone()
        };
        match perception_backend_effective(config) {
            PerceptionRolloutBackend::Dense => dense_perception_batch(&feature_x, &feature_s, config),
            PerceptionRolloutBackend::TiledAdjoint => {
                perception_tiled_adjoint_batch(feature_x, feature_s, config)
            }
            PerceptionRolloutBackend::Auto => unreachable!("auto perception backend resolved"),
        }
    }

    pub(super) fn perception_backend_effective(
        config: DirectBasisTrainConfig,
    ) -> PerceptionRolloutBackend {
        match config.perception_backend {
            PerceptionRolloutBackend::Auto => perception_backend_auto(config),
            PerceptionRolloutBackend::Dense => PerceptionRolloutBackend::Dense,
            PerceptionRolloutBackend::TiledAdjoint => PerceptionRolloutBackend::TiledAdjoint,
        }
    }

    pub(super) fn perception_backend_auto(config: DirectBasisTrainConfig) -> PerceptionRolloutBackend {
        if PERCEPTION_CUBE_ENABLED {
            if config.rollout_particles >= 128 {
                PerceptionRolloutBackend::TiledAdjoint
            } else {
                PerceptionRolloutBackend::Dense
            }
        } else {
            PerceptionRolloutBackend::Dense
        }
    }

    #[derive(Clone, Debug)]
    pub(super) struct PerceptionPreparedState {
        density: Tensor2Inner,
        offsets: Tensor2IntInner,
        permutation: Tensor2IntInner,
        raw_state_gradient: Tensor4Inner,
        state_gradient_inverse: Tensor3Inner,
    }

    #[derive(Clone, Debug)]
    pub(super) struct PerceptionAdjointState {
        x: Tensor3Inner,
        s: Tensor3Inner,
        prepared: Option<PerceptionPreparedState>,
        batch_size: usize,
        particle_count: usize,
        state_dims: usize,
        grid_eps: f32,
    }

    #[derive(Clone, Copy, Debug)]
    pub(super) struct PerceptionAdjointOp;

    impl Backward<InnerBackend, 2> for PerceptionAdjointOp {
        type State = PerceptionAdjointState;

        fn backward(
            self,
            ops: Ops<Self::State, 2>,
            grads: &mut Gradients,
            _checkpointer: &mut burn::backend::autodiff::checkpoint::base::Checkpointer,
        ) {
            let [x_parent, s_parent] = ops.parents;
            if x_parent.is_none() && s_parent.is_none() {
                return;
            }
            let feature_grad = grads.consume::<InnerBackend>(&ops.node);
            let feature_grad_tensor =
                Tensor::<InnerBackend, 3>::from_primitive(TensorPrimitive::Float(feature_grad));
            let device = feature_grad_tensor.device();

            #[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
            {
                let cube_config = perception_cube_adjoint_config(
                    ops.state.grid_eps,
                    x_parent.is_some(),
                    s_parent.is_some(),
                );
                let prepared_adjoint = ops.state.prepared.as_ref().and_then(|prepared| {
                    InnerBackend::perception_cube_adjoint_prepared(
                        ops.state.x.clone(),
                        ops.state.s.clone(),
                        feature_grad_tensor.clone(),
                        prepared.density.clone(),
                        prepared.offsets.clone(),
                        prepared.permutation.clone(),
                        prepared.raw_state_gradient.clone(),
                        prepared.state_gradient_inverse.clone(),
                        cube_config,
                    )
                });
                let device_adjoint = prepared_adjoint.or_else(|| {
                    InnerBackend::perception_cube_adjoint(
                        ops.state.x.clone(),
                        ops.state.s.clone(),
                        feature_grad_tensor.clone(),
                        cube_config,
                    )
                });
                if let Some(device_adjoint) = device_adjoint {
                    PERCEPTION_CUBE_ADJOINT_DEVICE_HITS.fetch_add(1, Ordering::Relaxed);
                    if ops.state.prepared.is_some() {
                        PERCEPTION_CUBE_PREPARED_REUSE_HITS.fetch_add(1, Ordering::Relaxed);
                    }
                    let device_adjoint =
                        device_adjoint.unwrap_or_else(|err| panic!("perception cube adjoint failed: {err}"));
                    if let Some(parent) = x_parent {
                        grads.register::<InnerBackend>(
                            parent.id,
                            device_adjoint.position_grad.into_primitive().tensor(),
                        );
                    }
                    if let Some(parent) = s_parent {
                        grads.register::<InnerBackend>(
                            parent.id,
                            device_adjoint.state_grad.into_primitive().tensor(),
                        );
                    }
                    return;
                }
            }

            PERCEPTION_CUBE_ADJOINT_FALLBACK_HITS.fetch_add(1, Ordering::Relaxed);
            let feature_grad = feature_grad_tensor
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_else(|err| panic!("perception adjoint readback failed: {err}"));
            let x_values = ops
                .state
                .x
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_else(|err| panic!("perception adjoint position readback failed: {err}"));
            let states = ops
                .state
                .s
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or_else(|err| panic!("perception adjoint state readback failed: {err}"));
            let positions = xy_positions_to_reference_positions(&x_values);
            let grid = perception_reference_grid(ops.state.grid_eps);
            let options = perception_reference_options(ops.state.grid_eps);
            let adjoint = burn_automata_kernels::perceive_adjoint_with_options(
                &positions,
                &states,
                ops.state.batch_size,
                ops.state.particle_count,
                ops.state.state_dims,
                &grid,
                options,
                &feature_grad,
            )
            .unwrap_or_else(|err| panic!("perception adjoint failed: {err}"));

            if let Some(parent) = x_parent {
                let position_grad = adjoint
                    .position
                    .iter()
                    .flat_map(|value| [value[0], value[1]])
                    .collect::<Vec<_>>();
                let tensor = Tensor::<InnerBackend, 3>::from_data(
                    TensorData::new(
                        position_grad,
                        [ops.state.batch_size, ops.state.particle_count, 2],
                    ),
                    &device,
                )
                .into_primitive()
                .tensor();
                grads.register::<InnerBackend>(parent.id, tensor);
            }
            if let Some(parent) = s_parent {
                let tensor = Tensor::<InnerBackend, 3>::from_data(
                    TensorData::new(
                        adjoint.state,
                        [
                            ops.state.batch_size,
                            ops.state.particle_count,
                            ops.state.state_dims,
                        ],
                    ),
                    &device,
                )
                .into_primitive()
                .tensor();
                grads.register::<InnerBackend>(parent.id, tensor);
            }
        }
    }

    pub(super) fn perception_tiled_adjoint_batch(
        x: Tensor3,
        s: Tensor3,
        config: DirectBasisTrainConfig,
    ) -> Tensor3 {
        let dims = s.shape().dims::<3>();
        let batch_size = dims[0];
        let particle_count = dims[1];
        let state_dims = dims[2];
        let x_dims = x.shape().dims::<3>();
        assert_eq!(
            x_dims,
            [batch_size, particle_count, 2],
            "perception tiled-adjoint expects x shape [batch, particles, 2]"
        );
        let x_primitive = x.into_primitive().tensor();
        let s_primitive = s.into_primitive().tensor();
        let x_inner = Tensor::<InnerBackend, 3>::from_primitive(TensorPrimitive::Float(
            x_primitive.primitive.clone(),
        ));
        let s_inner = Tensor::<InnerBackend, 3>::from_primitive(TensorPrimitive::Float(
            s_primitive.primitive.clone(),
        ));
        #[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
        let (output, prepared) = {
            let cube_config = perception_cube_adjoint_config(
                config.grid_eps,
                !config.stopgrad_pos,
                !config.stopgrad_state,
            );
            let prepared_forward = (config.stopgrad_pos && particle_count >= 512)
                .then(|| {
                    InnerBackend::perception_cube_forward_prepared(
                        x_inner.clone(),
                        s_inner.clone(),
                        cube_config,
                    )
                })
                .flatten();
            if let Some(device_forward) = prepared_forward {
                PERCEPTION_CUBE_FORWARD_DEVICE_HITS.fetch_add(1, Ordering::Relaxed);
                let device_forward = device_forward
                    .unwrap_or_else(|err| panic!("prepared perception cube forward failed: {err}"));
                let output = device_forward.features.into_primitive().tensor();
                let prepared = PerceptionPreparedState {
                    density: device_forward.density,
                    offsets: device_forward.offsets,
                    permutation: device_forward.permutation,
                    raw_state_gradient: device_forward.raw_state_gradient,
                    state_gradient_inverse: device_forward.state_gradient_inverse,
                };
                (output, Some(prepared))
            } else if let Some(device_forward) = InnerBackend::perception_cube_forward(
                x_inner.clone(),
                s_inner.clone(),
                cube_config,
            ) {
                PERCEPTION_CUBE_FORWARD_DEVICE_HITS.fetch_add(1, Ordering::Relaxed);
                (
                    device_forward
                        .unwrap_or_else(|err| panic!("perception cube forward failed: {err}"))
                        .features
                        .into_primitive()
                        .tensor(),
                    None,
                )
            } else {
                PERCEPTION_CUBE_FORWARD_FALLBACK_HITS.fetch_add(1, Ordering::Relaxed);
                (
                    dense_perception_batch_inner(&x_inner, &s_inner, config)
                        .into_primitive()
                        .tensor(),
                    None,
                )
            }
        };
        #[cfg(not(any(feature = "backend_wgpu", feature = "backend_cuda")))]
        let (output, prepared) = (
            dense_perception_batch_inner(&x_inner, &s_inner, config)
                .into_primitive()
                .tensor(),
            None,
        );
        let state = PerceptionAdjointState {
            x: x_inner,
            s: s_inner,
            prepared,
            batch_size,
            particle_count,
            state_dims,
            grid_eps: config.grid_eps,
        };
        let prep = PerceptionAdjointOp
            .prepare::<NoCheckpointing>([x_primitive.node.clone(), s_primitive.node.clone()])
            .compute_bound();
        let output = match prep.stateful() {
            OpsKind::Tracked(prep) => prep.finish(state, output),
            OpsKind::UnTracked(prep) => prep.finish(output),
        };
        Tensor::<BurnBackend, 3>::from_primitive(TensorPrimitive::Float(output))
    }

    pub(super) fn xy_positions_to_reference_positions(values: &[f32]) -> Vec<[f32; 4]> {
        values
            .chunks_exact(2)
            .map(|chunk| [chunk[0], chunk[1], 0.0, 0.0])
            .collect()
    }

    pub(super) fn perception_reference_grid(grid_eps: f32) -> burn_automata_kernels::HashGridConfig {
        let mut grid = crate::upstream_growing_2d_hashgrid();
        grid.eps = grid_eps.max(EPSILON);
        grid
    }

    pub(super) fn perception_reference_options(_grid_eps: f32) -> burn_automata_kernels::PerceptionOptions {
        let npa = NpaConfig::growing_2d();
        burn_automata_kernels::PerceptionOptions {
            state_grad: npa.state_grad,
            density_grad: npa.density_grad,
            eps0: npa.eps0.max(f32::MIN_POSITIVE),
            scale_equivariance: npa.scale_equivariant(),
            particle_density_equivariance: npa.particle_density_equivariant(),
            log_norm_grad: npa.log_norm_grad,
            log_norm_density_grad: npa.log_norm_density_grad,
            hybrid_state_gradient: true,
            position_features: npa.position_features,
        }
    }

    #[cfg(any(feature = "backend_wgpu", feature = "backend_cuda"))]
    pub(super) fn perception_cube_adjoint_config(
        grid_eps: f32,
        compute_position_grad: bool,
        compute_state_grad: bool,
    ) -> PerceptionCubeAdjointConfig {
        let npa = NpaConfig::growing_2d();
        PerceptionCubeAdjointConfig {
            eps: grid_eps.max(EPSILON),
            eps0: npa.eps0.max(f32::MIN_POSITIVE),
            state_grad: npa.state_grad,
            density_grad: npa.density_grad,
            scale_equivariance: npa.scale_equivariant(),
            particle_density_equivariance: npa.particle_density_equivariant(),
            log_norm_grad: npa.log_norm_grad,
            log_norm_density_grad: npa.log_norm_density_grad,
            hybrid_state_gradient: true,
            position_features: npa.position_features,
            compute_position_grad,
            compute_state_grad,
            grid_width: 16,
            grid_height: 16,
            sparse_grid_min_particles: 512,
        }
    }

    pub(super) fn dense_perception(x: &Tensor2, s: &Tensor2, config: DirectBasisTrainConfig) -> Tensor2 {
        let dims = s.shape().dims::<2>();
        let rows = dims[0];
        let state_dims = dims[1];
        let density = dense_particle_density(x, config);
        let chunk_size = dense_query_chunk_size(1, rows, state_dims, config.max_dense_chunk_floats);
        let mut chunks = Vec::new();
        for (start, len) in chunks_for(rows, chunk_size) {
            chunks.push(dense_perception_chunk(x, s, &density, config, start, len));
        }
        Tensor::cat(chunks, 0)
    }

    pub(super) fn dense_perception_chunk(
        x: &Tensor2,
        s: &Tensor2,
        density: &Tensor2,
        config: DirectBasisTrainConfig,
        start: usize,
        len: usize,
    ) -> Tensor2 {
        let dims = s.shape().dims::<2>();
        let rows = dims[0];
        let state_dims = dims[1];
        let xi = x
            .clone()
            .narrow(0, start, len)
            .unsqueeze_dim::<3>(1)
            .expand([len, rows, 2]);
        let xj = x.clone().unsqueeze_dim::<3>(0).expand([len, rows, 2]);
        let diff = xj - xi;
        let dist2 = diff
            .clone()
            .mul(diff.clone())
            .sum_dim(2)
            .squeeze_dim::<2>(2);
        let eps = config.grid_eps.max(EPSILON);
        let compact = relu(dist2.clone().mul_scalar(-1.0).add_scalar(eps * eps));
        let compact2 = compact.clone().mul(compact.clone());
        let smooth = compact2
            .mul(compact)
            .mul_scalar(4.0 / (std::f32::consts::PI * eps.powi(8)));
        let volume_j = density.clone().transpose().recip().expand([len, rows]);
        let blur = smooth.clone().mul(volume_j.clone()).matmul(s.clone());

        let r = dist2.add_scalar(EPSILON * EPSILON).sqrt();
        let spiky = relu(r.clone().mul_scalar(-1.0).add_scalar(eps));
        let spiky_mag = spiky
            .clone()
            .mul(spiky)
            .div(r)
            .mul_scalar(30.0 / (std::f32::consts::PI * eps.powi(5)));
        let grad = diff
            .clone()
            .mul(spiky_mag.unsqueeze_dim::<3>(2).expand([len, rows, 2]));
        let density_grad = log_normalize_vectors(
            grad.clone()
                .sum_dim(1)
                .squeeze_dim::<2>(1)
                .mul_scalar((eps / 0.1).powi(3) / rows.max(1) as f32),
        );

        let sj = s
            .clone()
            .unsqueeze_dim::<3>(0)
            .expand([len, rows, state_dims]);
        let si = s
            .clone()
            .narrow(0, start, len)
            .unsqueeze_dim::<3>(1)
            .expand([len, rows, state_dims]);
        let state_diff = sj - si;
        let volume_grad = grad.mul(volume_j.unsqueeze_dim::<3>(2).expand([len, rows, 2]));
        let state_grad = state_diff
            .unsqueeze_dim::<4>(3)
            .expand([len, rows, state_dims, 2])
            .mul(
                volume_grad
                    .clone()
                    .unsqueeze_dim::<4>(2)
                    .expand([len, rows, state_dims, 2]),
            )
            .sum_dim(1)
            .squeeze_dim::<3>(1);
        let state_grad = apply_moment_correction_2d(state_grad, diff, volume_grad);
        let state_grad = log_normalize_state_gradient(state_grad);

        Tensor::cat(
            vec![
                s.clone().narrow(0, start, len),
                blur,
                state_grad,
                density_grad,
            ],
            1,
        )
    }

    pub(super) fn dense_perception_batch(x: &Tensor3, s: &Tensor3, config: DirectBasisTrainConfig) -> Tensor3 {
        let dims = s.shape().dims::<3>();
        let batches = dims[0];
        let rows = dims[1];
        let state_dims = dims[2];
        let density = dense_particle_density_batch(x, config);
        let chunk_size =
            dense_query_chunk_size(batches, rows, state_dims, config.max_dense_chunk_floats);
        let mut chunks = Vec::new();
        for (start, len) in chunks_for(rows, chunk_size) {
            chunks.push(dense_perception_batch_chunk(
                x, s, &density, config, start, len,
            ));
        }
        Tensor::cat(chunks, 1)
    }

    pub(super) fn dense_perception_batch_chunk(
        x: &Tensor3,
        s: &Tensor3,
        density: &Tensor3,
        config: DirectBasisTrainConfig,
        start: usize,
        len: usize,
    ) -> Tensor3 {
        let dims = s.shape().dims::<3>();
        let batches = dims[0];
        let rows = dims[1];
        let state_dims = dims[2];
        let xi = x
            .clone()
            .narrow(1, start, len)
            .unsqueeze_dim::<4>(2)
            .expand([batches, len, rows, 2]);
        let xj = x
            .clone()
            .unsqueeze_dim::<4>(1)
            .expand([batches, len, rows, 2]);
        let diff = xj - xi;
        let dist2 = diff
            .clone()
            .mul(diff.clone())
            .sum_dim(3)
            .squeeze_dim::<3>(3);
        let eps = config.grid_eps.max(EPSILON);
        let compact = relu(dist2.clone().mul_scalar(-1.0).add_scalar(eps * eps));
        let compact2 = compact.clone().mul(compact.clone());
        let smooth = compact2
            .mul(compact)
            .mul_scalar(4.0 / (std::f32::consts::PI * eps.powi(8)));
        let volume_j = density
            .clone()
            .swap_dims(1, 2)
            .recip()
            .expand([batches, len, rows]);
        let blur = smooth.clone().mul(volume_j.clone()).matmul(s.clone());

        let r = dist2.add_scalar(EPSILON * EPSILON).sqrt();
        let spiky = relu(r.clone().mul_scalar(-1.0).add_scalar(eps));
        let spiky_mag = spiky
            .clone()
            .mul(spiky)
            .div(r)
            .mul_scalar(30.0 / (std::f32::consts::PI * eps.powi(5)));
        let grad = diff.clone().mul(
            spiky_mag
                .unsqueeze_dim::<4>(3)
                .expand([batches, len, rows, 2]),
        );
        let density_grad = log_normalize_vectors_batch(
            grad.clone()
                .sum_dim(2)
                .squeeze_dim::<3>(2)
                .mul_scalar((eps / 0.1).powi(3) / rows.max(1) as f32),
        );

        let sj = s
            .clone()
            .unsqueeze_dim::<4>(1)
            .expand([batches, len, rows, state_dims]);
        let si = s
            .clone()
            .narrow(1, start, len)
            .unsqueeze_dim::<4>(2)
            .expand([batches, len, rows, state_dims]);
        let state_diff = sj - si;
        let volume_grad = grad.mul(
            volume_j
                .unsqueeze_dim::<4>(3)
                .expand([batches, len, rows, 2]),
        );
        let state_grad = state_diff
            .unsqueeze_dim::<5>(4)
            .expand([batches, len, rows, state_dims, 2])
            .mul(
                volume_grad
                    .clone()
                    .unsqueeze_dim::<5>(3)
                    .expand([batches, len, rows, state_dims, 2]),
            )
            .sum_dim(2)
            .squeeze_dim::<4>(2);
        let state_grad = apply_moment_correction_2d_batch(state_grad, diff, volume_grad);
        let state_grad = log_normalize_state_gradient_batch(state_grad);

        Tensor::cat(
            vec![
                s.clone().narrow(1, start, len),
                blur,
                state_grad,
                density_grad,
            ],
            2,
        )
    }

    pub(super) fn dense_perception_batch_inner(
        x: &Tensor3Inner,
        s: &Tensor3Inner,
        config: DirectBasisTrainConfig,
    ) -> Tensor3Inner {
        dense_perception_batch_generic::<InnerBackend>(x, s, config)
    }

    pub(super) fn dense_perception_batch_generic<B: burn::tensor::backend::Backend>(
        x: &Tensor<B, 3>,
        s: &Tensor<B, 3>,
        config: DirectBasisTrainConfig,
    ) -> Tensor<B, 3> {
        let dims = s.shape().dims::<3>();
        let batches = dims[0];
        let rows = dims[1];
        let state_dims = dims[2];
        let density = dense_particle_density_batch_generic(x, config);
        let chunk_size =
            dense_query_chunk_size(batches, rows, state_dims, config.max_dense_chunk_floats);
        let mut chunks = Vec::new();
        for (start, len) in chunks_for(rows, chunk_size) {
            chunks.push(dense_perception_batch_chunk_generic(
                x, s, &density, config, start, len,
            ));
        }
        Tensor::cat(chunks, 1)
    }

    pub(super) fn dense_perception_batch_chunk_generic<B: burn::tensor::backend::Backend>(
        x: &Tensor<B, 3>,
        s: &Tensor<B, 3>,
        density: &Tensor<B, 3>,
        config: DirectBasisTrainConfig,
        start: usize,
        len: usize,
    ) -> Tensor<B, 3> {
        let dims = s.shape().dims::<3>();
        let batches = dims[0];
        let rows = dims[1];
        let state_dims = dims[2];
        let xi = x
            .clone()
            .narrow(1, start, len)
            .unsqueeze_dim::<4>(2)
            .expand([batches, len, rows, 2]);
        let xj = x
            .clone()
            .unsqueeze_dim::<4>(1)
            .expand([batches, len, rows, 2]);
        let diff = xj - xi;
        let dist2 = diff
            .clone()
            .mul(diff.clone())
            .sum_dim(3)
            .squeeze_dim::<3>(3);
        let eps = config.grid_eps.max(EPSILON);
        let compact = relu(dist2.clone().mul_scalar(-1.0).add_scalar(eps * eps));
        let compact2 = compact.clone().mul(compact.clone());
        let smooth = compact2
            .mul(compact)
            .mul_scalar(4.0 / (std::f32::consts::PI * eps.powi(8)));
        let volume_j = density
            .clone()
            .swap_dims(1, 2)
            .recip()
            .expand([batches, len, rows]);
        let blur = smooth.clone().mul(volume_j.clone()).matmul(s.clone());

        let r = dist2.add_scalar(EPSILON * EPSILON).sqrt();
        let spiky = relu(r.clone().mul_scalar(-1.0).add_scalar(eps));
        let spiky_mag = spiky
            .clone()
            .mul(spiky)
            .div(r)
            .mul_scalar(30.0 / (std::f32::consts::PI * eps.powi(5)));
        let grad = diff.clone().mul(
            spiky_mag
                .unsqueeze_dim::<4>(3)
                .expand([batches, len, rows, 2]),
        );
        let density_grad = log_normalize_vectors_batch_generic(
            grad.clone()
                .sum_dim(2)
                .squeeze_dim::<3>(2)
                .mul_scalar((eps / 0.1).powi(3) / rows.max(1) as f32),
        );

        let sj = s
            .clone()
            .unsqueeze_dim::<4>(1)
            .expand([batches, len, rows, state_dims]);
        let si = s
            .clone()
            .narrow(1, start, len)
            .unsqueeze_dim::<4>(2)
            .expand([batches, len, rows, state_dims]);
        let state_diff = sj - si;
        let volume_grad = grad.mul(
            volume_j
                .unsqueeze_dim::<4>(3)
                .expand([batches, len, rows, 2]),
        );
        let state_grad = state_diff
            .unsqueeze_dim::<5>(4)
            .expand([batches, len, rows, state_dims, 2])
            .mul(
                volume_grad
                    .clone()
                    .unsqueeze_dim::<5>(3)
                    .expand([batches, len, rows, state_dims, 2]),
            )
            .sum_dim(2)
            .squeeze_dim::<4>(2);
        let state_grad =
            apply_moment_correction_2d_batch_generic::<B>(state_grad, diff, volume_grad);
        let state_grad = log_normalize_state_gradient_batch_generic(state_grad);

        Tensor::cat(
            vec![
                s.clone().narrow(1, start, len),
                blur,
                state_grad,
                density_grad,
            ],
            2,
        )
    }
