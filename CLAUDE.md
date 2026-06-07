# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This is a **greenfield project**. As of this writing the directory contains only `pv_fault_classification_context.md` — the authoritative project spec — and no source code. Future instances should:

1. Read `pv_fault_classification_context.md` in full before making architectural decisions. It contains the physics, dataset details, target metrics, ablation table structure, recommended file layout, and pitfalls.
2. When code starts existing, prefer extending the structure proposed in the spec (`src/`, `notebooks/`, `configs/`, `data/`) rather than inventing a new layout.

## What this project is

Physics-guided fault classifier for the **InfraredSolarModules** dataset (20,000 IR images, 24×40 px, 12 classes). The single novel contribution — and the reason this project exists — is **IEC 62446-3 ΔT normalization** applied as a learned preprocessing step before classification. Every prior open-source baseline trains on raw thermal pixels, which conflates fault signal with irradiance. This project infers irradiance from healthy-panel statistics and feeds a residual map to the classifier instead.

Headline targets to beat: Soiling F1 from ~0.30 → >0.65, macro F1 from ~0.73 → >0.85, **without dropping any of the 11 fault classes** from evaluation.

## Architecture (three stages, must stay decoupled)

1. **Baseline temperature estimator** (`src/estimator.py`) — small CNN / MobileNetV3-Small with a scalar regression head, MAE loss. Predicts what a healthy panel's mean temperature should look like under the current (unknown) irradiance.
2. **Residual generator** (`src/residual.py`) — deterministic, no learned parameters: `residual = raw − estimator(raw)`, optionally normalized by baseline. No gradients flow through this step during classifier training.
3. **Hierarchical classifier** (`src/model.py`) — EfficientNet-B0 backbone on the residual map. Heads: binary (healthy/fault) → 2-class gate (environmental vs electrical) → 3-class environmental head (Soiling/Shadowing/Vegetation) + 8-class electrical head (Cell, Cell-Multi, Hot-Spot, Hot-Spot-Multi, Diode, Diode-Multi, Cracking, Offline-Module). Soft routing via temperature-scaled gate.

## Load-bearing constraints (do not violate without discussing first)

These are not style preferences. Each is a methodological choice the paper depends on; breaking them silently invalidates the contribution.

- **The baseline estimator must be trained only on No-Anomaly samples.** Any fault leakage teaches it to suppress fault signatures in the residual, defeating the entire project. Enforce this with an assertion in the estimator's dataloader — not just a comment.
- **Thermal MixUp must mix same-class pairs only.** A blend of a Soiling image and a Diode image has no real-world analog. The label stays unchanged because both inputs share it.
- **Macro F1 is the headline metric**, not weighted F1 or overall accuracy. Weighted F1 is dominated by Cell and will look fine even when Soiling F1 = 0. Always report macro F1 + per-class F1, and call Soiling F1 out explicitly.
- **No random crops, no color jitter.** 24×40 is too small to crop without destroying spatial structure; color jitter is meaningless on a single-channel thermal image. Permitted augmentations: hflip, vflip, ±10% brightness jitter, same-class thermal MixUp.
- **Resize 24×40 → 128×128 with bicubic or bilinear, never nearest-neighbor.** Nearest creates blocking artifacts the CNN will learn as spurious features.
- **Evaluation split is fixed:** fault-only 10k → 7k/1.5k/1.5k stratified, matching the 2025 benchmark paper. Full 20k is reported separately for binary results only. Once `data/splits/*.csv` exist, do not regenerate them.

## Class index (must stay consistent across estimator, splits, and classifier)

A mismatch here produces silent, catastrophic bugs. The canonical mapping from the spec:

```
No-Anomaly: 0, Cell: 1, Cell-Multi: 2, Cracking: 3, Hot-Spot: 4, Hot-Spot-Multi: 5,
Shadowing: 6, Diode: 7, Diode-Multi: 8, Vegetation: 9, Soiling: 10, Offline-Module: 11
```

Physical grouping for the hierarchical head:
- Environmental: Soiling, Shadowing, Vegetation
- Electrical: Cell, Cell-Multi, Hot-Spot, Hot-Spot-Multi, Diode, Diode-Multi, Cracking, Offline-Module

## Dataset

Source: https://github.com/RaptorMaps/InfraredSolarModules (Kaggle mirror also available). Expected on disk at `data/images/*.jpg` with `data/module_metadata.json` mapping `image_number → {image_filepath, anomaly_class}`. Verify counts on first load — they should match the spec's table exactly; mismatches mean a corrupted download.

## Commands

No build/test/lint commands exist yet — there is no code. When the project is bootstrapped, training and evaluation will live behind `src/train.py` and `src/evaluate.py` per the spec's recommended layout. Add the exact invocations here once they exist.

## Dependencies (planned, per spec)

`torch>=2.1`, `torchvision>=0.16`, `timm>=0.9` (EfficientNet/Swin/ViT), `scikit-learn>=1.3` (metrics + stratified splits), `grad-cam>=1.4` (explainability), plus the usual `numpy`/`Pillow`/`pandas`/`matplotlib`/`seaborn`/`pyyaml`/`tqdm`.

## Evaluation deliverables (paper-driven)

Anything claiming "done" on the modeling work must produce:
- Per-class precision/recall/F1 + macro F1 + weighted F1 + full 12-class confusion matrix + one-vs-rest ROC-AUC.
- The ablation table from the spec (raw vs residual × flat vs hierarchical, then +focal loss, +thermal MixUp).
- GradCAM panels per class: raw | baseline-model CAM | residual | your-model CAM on residual. If activations land on image corners/borders for any class, the model learned camera artifacts — fix before claiming the result.

## User context

Built by Abdul Moiz (final-year EE, GIK Institute). Power electronics background, newer to deep learning tooling. Lean on physics intuition when justifying architecture; explain PyTorch/timm/GradCAM moving parts as they come up rather than assuming familiarity.
