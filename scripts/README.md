# Repository Scripts

Scripts are thin operational wrappers around Rust/Burn capabilities. They do
not implement training objectives, experiment matrices, or paper rendering.

## Layout

| Directory | Scope |
| --- | --- |
| [`ci/`](ci/) | Hosted/manual compile, feature, and benchmark entrypoints |
| [`web/`](web/) | WebAssembly build, runtime validation, and model packaging |
| [`reference/selforg/`](reference/selforg/) | Pinned upstream checkout, fixture/catalog import, and independent parity oracle |
| [`reference/dino/`](reference/dino/) | One-time official DINOv2 weight conversion |
| [`validation/`](validation/) | Hardware-specific GPU and experimental 3D validation |

Python is restricted to external model interchange and independent reference
checks. New trainers, benchmarks, report generators, and experiment sweeps must
be Rust CLI commands driven by TOML under `configs/`.

## Common Commands

```bash
scripts/ci/check_inference_features.sh
scripts/web/build_wasm.sh
node scripts/web/validate_web_runtime.mjs --static

scripts/reference/selforg/fetch_selforg_npa.sh
scripts/reference/selforg/fetch_selforg_npa_targets.sh
python3 scripts/reference/selforg/export_selforg_npa_fixture.py --help
python3 scripts/reference/selforg/validate_import_parity.py --help

REQUIRE_BPK=1 scripts/validation/validate_gpu_e2e.sh
python3 scripts/validation/3d/validate_3d_catalog.py --help
```

All entrypoints assume the repository root is the current working directory.
Generated files belong under ignored `target/`, `artifacts/`, `models/`,
`.cache/`, or `data/` directories.
