# Scripts

Python in this repository is limited to external interchange and reference
validation:

- `import_selforg_catalog.py` imports the external SelfOrg catalog.
- `export_npa_checkpoint.py` exports unsupported PyTorch checkpoint variants.
- `setup_dino_vits.py` creates the Burn DINO model pack from a Torch checkpoint.
- `validate_*`, `compare_3d_candidate.py`, and `catalog3d_validation/` provide
  parity/reference checks for imported models and renderer candidates.

Do not add Python training, benchmark-matrix, or paper-rendering entrypoints
here. New experiments should be Rust/Burn CLI commands with TOML configs.
