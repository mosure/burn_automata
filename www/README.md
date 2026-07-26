# burn_automata web

The generated Bevy/WebGPU application is hosted directly from this directory.

Run `scripts/build_wasm.sh` to package the viewer and isolated Burn training
worker into ignored `pkg/` directories. The Pages workflow downloads the
versioned model bundle, verifies every asset digest, builds both Wasm modules,
and deploys this directory.

`scripts/validate_web_runtime.mjs` is the deployment gate. It opens the Bevy
viewer in Chrome, exercises image selection, the Train/Stop UI, conditioned
DINO/HyperNPA inference, and bounded fixed and adaptive Target2D WGPU jobs in
the dedicated worker. Browser training uses the same trainer and objective as
native training with a 256-particle, batch-one, 8-16-step bounded session
profile so rendering and optimization can safely share a browser GPU. The
Pages workflow runs the smoke against the local artifact and again against the
deployed URL.
