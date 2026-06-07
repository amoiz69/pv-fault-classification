# PV Thermal Fault Classification

Physics-guided irradiance-normalized fault classifier for the InfraredSolarModules dataset.
See `pv_fault_classification_context.md` for the full project spec and `CLAUDE.md` for
guidance to AI assistants working in this repo.

## Setup (Phase 0)

```bash
# 1. Create conda env
conda create -n pv-fault python=3.11 -y
conda activate pv-fault
pip install -r requirements.txt

# 2. Download dataset into data/
#    Expected layout afterward:
#      data/images/*.jpg          (20,000 files, 24x40)
#      data/module_metadata.json
python scripts/download_dataset.py   # or follow manual instructions below

# 3. Sanity check
python scripts/verify_dataset.py
```

Manual download: clone https://github.com/RaptorMaps/InfraredSolarModules and copy
`images/` and `module_metadata.json` into `data/`.

## Layout

```
src/         training/eval code (dataset, estimator, residual, model, losses, ...)
scripts/     one-off CLIs (verify_dataset, download, ...)
notebooks/   EDA, residual analysis, ablation
configs/     YAML hyperparameter configs
data/        labels (module_metadata.json) + images/ (gitignored)
checkpoints/ trained weights (gitignored)
```
