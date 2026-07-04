# Hyper2D Direct-Basis Experiments

Run a bundled experiment with:

```sh
cargo run -p burn_automata --features cli -- train-hyper2d-direct-basis --config configs/hyper2d_direct_basis/omnisvg_1k.toml
```

The TOML file is the experiment recipe. Values supplied in TOML take precedence over the flat CLI flags for the same setting, while omitted values keep the existing CLI defaults.

The OmniSVG recipes default to `download = false` so they use the local cache. Set `[source.omnisvg].download = true` when the cache needs to be populated or refreshed.

These configs intentionally use `[gpu].backend = "burn-wgpu"`. The older
upstream Python/CUDA scripts remain useful for parity checks and historical
comparison, but new 2D direct-basis experiments should start from these
Burn/WGPU TOML recipes.
