# Implementation Plan

## Components

1. `burn_automata_kernels`
   - Optimized CPU oracle for normalized SPH density, blur, hybrid state gradients, density gradients, hashgrid layout, splatting, and Gaussian decoding.
   - Future CubeCL/WGPU kernels must be validated against these functions and the import parity script.

2. `burn_automata`
   - Model config and weight schema.
   - Rollout and training APIs.
   - Checksumed BPK import/export plus direct upstream `.pth` import.
   - CLI smoke and benchmark entry points.
   - Direct WGPU linked-grid inference executor for GPU hashgrid, perception, MLP update, and Euler integration.

3. `bevy_automata`
   - Live rollout viewer.
   - BSN UI tree for simulation controls, model status, and backward probe status.
   - Optional Gaussian splatting plugin integration.
   - `bevy_burn` bridge hooks for GPU buffer ownership through Bevy `ShaderBuffer` assets.

## Milestones

| milestone | validation |
| --- | --- |
| CPU oracle | deterministic unit tests, shape checks, finite outputs |
| CLI baseline | manifest, import, inference, train, benchmark commands |
| Viewer baseline | Bevy viewer check with and without splatting feature |
| GPU feature surface | Burn WGPU compile check, direct WGPU compile check, and CPU/GPU golden comparisons on fixed seeds |
| Upstream import | direct checkpoint import tests, parameter count checks, one-step parity script |
| E2E training | supervised loss reduction tests and future rendered validation snapshots |
