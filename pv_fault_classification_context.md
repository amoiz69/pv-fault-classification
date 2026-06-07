# PV Thermal Fault Classification — Project Context

## One-line summary
Physics-guided irradiance-normalized fault classification on the InfraredSolarModules dataset, targeting the minority classes (Soiling, Hot-Spot, Diode-Multi) that every existing paper either drops or fails on.

---

## Problem statement

Every published model trained on InfraredSolarModules uses raw thermal pixel values as classifier input. This is fundamentally wrong. Solar panel temperature is a function of both fault state AND irradiance/ambient conditions. A soiling signature at 400 W/m² morning irradiance looks completely different from the same soiling at 1000 W/m² noon irradiance. Models trained on the raw mixture cannot separate fault signal from environmental noise, which is why soiling achieves F1 = 0.20–0.33 across all published baselines.

The fix — irradiance normalization before classification — is implemented proprietary by Raptor Maps (the dataset creators) in their commercial software. They normalize all temperature deltas to a reference of 1000 W/m² per IEC 62446-3. No published open-source work has done this. The dataset ships without irradiance metadata, so the normalization must be inferred from the images themselves.

**Target:** Bring soiling F1 from ~0.30 to >0.65. Bring overall multiclass accuracy above 85% without dropping any class.

---

## Dataset

**Name:** InfraredSolarModules  
**Source:** https://github.com/RaptorMaps/InfraredSolarModules  
**Kaggle mirror:** https://www.kaggle.com/datasets/marcosgabriel/infrared-solar-modules  
**Size:** 20,000 infrared images, each 24×40 pixels  
**Format:** JPEG images + `module_metadata.json` with class labels  
**License:** Open for research use  
**Origin:** 826 PV systems across 25 countries, collected by piloted aircraft and UAVs with midwave infrared cameras

### JSON structure
```json
{
  "<image_number>": {
    "image_filepath": "images/<image_number>.jpg",
    "anomaly_class": "<class_name>"
  }
}
```

### Class distribution (full 20,000 images)

| Class | Count | % of total | Notes |
|---|---|---|---|
| No-Anomaly | 10,000 | 50.0% | Healthy panels — majority class |
| Cell | 1,877 | 9.4% | Hotspot in single cell |
| Vegetation | 1,639 | 8.2% | Panels blocked by vegetation |
| Diode | 1,499 | 7.5% | Activated bypass diode (~1/3 module) |
| Cell-Multi | 1,288 | 6.4% | Hotspots in multiple cells |
| Shadowing | 1,056 | 5.3% | Obstructed by structures or adjacent rows |
| Cracking | 940 | 4.7% | Surface cracks |
| Offline-Module | 827 | 4.1% | Entire module heated |
| Hot-Spot | 249 | 1.2% | Square geometry hotspot, single cell |
| Hot-Spot-Multi | 246 | 1.2% | Square geometry hotspots, multiple cells |
| **Soiling** | **204** | **1.0%** | **Dirt/dust — hardest class, F1~0.30** |
| Diode-Multi | 175 | 0.9% | Multiple activated bypass diodes (~2/3 module) |

When papers report "fault-only" results they split the 10,000 fault images into 7,000 train / 1,500 val / 1,500 test using the natural class distribution. Do not oversample No-Anomaly into the fault-only split.

---

## Why the dataset is hard (root causes)

### 1. Severe class imbalance
No-Anomaly (10,000) vs. Soiling (204) is a 49:1 ratio. Standard cross-entropy loss is dominated by the majority class. Focal loss and class-weighted loss are necessary.

### 2. Missing irradiance metadata
The dataset was collected across different times of day, seasons, latitudes, and irradiance conditions. Raptor Maps stripped metadata before publishing. This means:
- A Soiling sample at dawn and a Soiling sample at noon have completely different absolute pixel values despite being the same fault type.
- Models learn irradiance-dependent texture instead of fault-specific thermal signatures.
- This is the core problem this project solves.

### 3. Inter-class visual similarity
Shadowing and Vegetation produce visually similar thermal patterns (blocked irradiance → cooler panels or activated bypass diodes). Cell and Hot-Spot differ mainly in spatial geometry. Diode and Shadowing both show ~1/3 module heating. A flat softmax head cannot easily separate these without physics guidance.

### 4. Low resolution
24×40 pixels. Standard augmentations (random crop, etc.) destroy spatial structure. Augmentation must preserve thermal gradient patterns.

### 5. Domain shift across plants
The dataset comes from 826 systems globally. A model trained on the full dataset implicitly learns plant-specific temperature distributions, not fault signatures. This is confirmed by UMAP analysis in published work — samples cluster by plant, not by class.

---

## Physics of each fault class (EE background required)

Understanding these is essential for building the residual normalization and the hierarchical classifier.

### Cell / Cell-Multi
A shunted or resistive cell forces the string current through a low-resistance path. Power dissipation: P = I² × R_shunt. Since string current I is proportional to irradiance G, ΔT scales roughly linearly with G. At G = 1000 W/m², ΔT is large and clearly detectable. At G = 300 W/m² (dawn), ΔT may be below the camera noise floor. Spatial signature: 1–3 cells clearly hotter than neighbors, localized and sharp-edged.

### Hot-Spot / Hot-Spot-Multi
Similar mechanism to Cell but specifically caused by reverse-biased cells operating in avalanche breakdown. The IEC standard distinguishes Hot-Spot by its square geometry (matching cell geometry) vs. Cell which can be irregularly shaped. ΔT ∝ G. Same irradiance dependence as Cell.

### Diode / Diode-Multi
A bypass diode activates when a substring is shaded, cracked, or has mismatched Isc. The diode conducts the full string current in reverse. Power dissipation: P = I_string × V_diode ≈ I_string × 0.7V. String current ∝ G, so ΔT ∝ G. Spatial signature: approximately 1/3 of the module (one substring) uniformly hotter. Sharp boundary at the bypass diode boundary. Very high ΔT at standard conditions (~20–40°C above neighbors).

### Cracking
Physical damage to the cell or encapsulant. Creates locally increased resistance. If the crack isolates a cell section, that section stops contributing to current generation and gets heated by the rest of the string. ΔT is moderate and somewhat irradiance-dependent. Signature is often irregular, following the crack geometry.

### Soiling
Dust, dirt, bird droppings, or debris on the module surface. Reduces effective irradiance reaching cells. Two thermal effects:
1. Soiled cells absorb less light → slightly cooler (if partially shaded effect dominates).
2. If soiling is thick enough, the soiled cells become current-limiting → bypass diode activates → slightly hotter region.
In practice, soiling produces a diffuse, low-magnitude ΔT (typically 2–5°C) that changes sign depending on soiling density and current operating point. This is WHY it is so hard: the signal is small, diffuse, and environmentally confounded. After irradiance normalization, the relative signature becomes more consistent.

### Shadowing
External obstruction (adjacent panel rows at low sun angle, mounting structures, trees) blocks irradiance on a subset of cells. Bypassed substring heats up. Almost identical spatial signature to Diode. The distinction: Shadowing is reversible and time-of-day dependent; Diode is persistent. In a static dataset you cannot distinguish them temporally, so the model must use spatial pattern differences.

### Vegetation
Panels physically covered by vegetation or debris from above. Full panels or substrings are blocked. Higher ΔT than Shadowing because more area is affected and vegetation can create partial shading patterns that are visually distinct.

### Offline-Module
The entire module is disconnected from the string or operating in open-circuit condition. With no current flowing, the module heats uniformly via direct solar absorption rather than converting light to electricity. Temperature above neighboring modules proportional to the fraction of irradiance that isn't being converted. Very high absolute temperature, uniform spatial distribution.

---

## The innovation — what no paper has done

### IEC 62446-3 ΔT normalization (the core contribution)
The IEC standard for PV inspection defines: normalize all anomaly temperature deltas to a reference irradiance of G_ref = 1000 W/m² using:

```
ΔT_normalized = ΔT_measured × (G_ref / G_measured)
```

This is what Raptor Maps does in their proprietary software. Since we don't have G_measured for the dataset images, we estimate it from the image itself.

**Irradiance proxy estimation:**  
The mean temperature of healthy (No-Anomaly) panels is a monotonic function of G × (1 - η) + T_ambient, where η is module efficiency (~0.18) and T_ambient is ambient temperature. Within a single flight, ambient temperature is nearly constant. So mean panel temperature ≈ f(G). We can learn this function from No-Anomaly samples and use it as a surrogate for G.

**Implementation:**  
1. Train a regression model on No-Anomaly crops to predict their mean pixel intensity (proxy for irradiance-adjusted baseline temperature).  
2. For each image (any class), compute: `residual = raw_image - predicted_baseline_map`  
3. Use residual maps as classifier input instead of raw images.

After this transform, a Soiling sample at dawn and a Soiling sample at noon should have similar residual signatures because the irradiance confound has been removed.

### Hierarchical classification (secondary contribution)
Rather than a flat 12-class softmax, split into two physically motivated groups:

**Environmental faults** (caused by external obstruction, irradiance reduction):
- Soiling, Shadowing, Vegetation

**Electrical faults** (caused by internal cell/component failure):
- Cell, Cell-Multi, Hot-Spot, Hot-Spot-Multi, Diode, Diode-Multi, Cracking, Offline-Module

Stage 1: Binary (healthy vs. fault)  
Stage 2: Environmental vs. electrical  
Stage 3: Fine-grained within each group  

This hierarchy mirrors how a PV engineer would triage a fault: first confirm it's real, then determine if it's an operations issue (clean it / remove shade) or a maintenance issue (replace the module).

---

## Architecture

### Stage 1 — Baseline temperature estimator

**Purpose:** Learn to predict what a healthy panel should look like at the current irradiance/environmental conditions.  
**Input:** 24×40 raw thermal crop  
**Output:** Single scalar — predicted mean temperature (proxy for baseline)  
**Training data:** No-Anomaly class only (10,000 samples)  
**Architecture:** Small CNN or MobileNetV3-Small regression head  
**Loss:** MAE  
**Key constraint:** Must NOT be trained on anomalous samples — it needs to learn the healthy baseline only.

### Stage 2 — Residual map generator

**Purpose:** Remove irradiance confound.  
**Process:**
```python
# pseudocode
baseline_temp = estimator(raw_image)          # scalar
baseline_map = torch.full_like(raw_image, baseline_temp)
residual = raw_image - baseline_map           # ΔT map
residual = residual / (baseline_temp + 1e-6)  # optional: normalize by baseline
```
**Note:** This is deterministic, not learned. No gradients flow through this step during classifier training.

### Stage 3 — Hierarchical fault classifier

**Input:** ΔT residual map (24×40, single channel)  
**Preprocessing:** Resize to 128×128 via bicubic (preserves thermal gradient structure better than bilinear). Normalize to [-1, 1] based on residual statistics (not [0,1] — residuals are signed).

**Binary head (Stage 3a):**  
Architecture: EfficientNet-B0 pretrained on ImageNet, fine-tuned  
Output: P(fault | residual)  
Loss: Focal loss with γ=2  

**Hierarchical head (Stage 3b, runs only if Stage 3a > threshold):**  
Two separate classification heads on the same backbone:
- Environmental branch: 3-class softmax (Soiling / Shadowing / Vegetation)  
- Electrical branch: 8-class softmax (Cell, Cell-Multi, HotSpot, HotSpot-Multi, Diode, Diode-Multi, Cracking, Offline)  
Gating: An additional 2-class head predicts Environmental vs. Electrical; soft routing via temperature scaling.

**Loss for Stage 3b:**  
```python
# class-weighted focal loss
alpha = 1.0 / class_frequency  # inverse frequency weighting
loss = focal_loss(logits, targets, alpha=alpha, gamma=2.0)
```

**Augmentation (thermal-aware):**  
- Horizontal/vertical flip: safe (thermal pattern is symmetric)
- Brightness jitter within ±10%: safe (simulates small irradiance variation, residual absorbs this)  
- Thermal MixUp: `mixed = λ × img_a + (1-λ) × img_b` where both images are the SAME class — this respects physical additivity of thermal signatures  
- Do NOT use: random crop (destroys 24×40 spatial structure), color jitter (meaningless for grayscale thermal)

---

## Baseline models to compare against

You must beat these baselines for the paper to be publishable:

| Model | Overall accuracy | Soiling F1 | Source |
|---|---|---|---|
| Random Forest | 84.0% | not reported | Raptor Maps 2025 paper |
| Custom CNN (7-layer) + SMOTE | 77.0% | ~0.30 | arXiv 2409.16069 |
| EfficientNet-B0 + SVM | 93.0% | drops minority classes | Duranay 2023 |
| ResNet-18 | ~88% | ~0.25 | multiple papers |
| Swin-Tiny | 73.0% multiclass | 0.20–0.33 | arXiv 2509.07039 |
| ViT-Tiny | 71.0% multiclass | similar | arXiv 2509.07039 |

**Critical note:** Papers that report 90%+ accuracy are typically using binary classification only or dropping the minority classes. Always verify what classes are included in the evaluation. Your evaluation protocol must include ALL 11 fault classes.

---

## Evaluation protocol

**Split:** Use the same split as the 2025 benchmarking paper for reproducibility:
- Fault-only 10,000 samples → 7,000 train / 1,500 val / 1,500 test, stratified
- Full 20,000 samples → report binary results separately

**Metrics to report (per class AND macro-averaged):**
- Precision, Recall, F1 per class
- Macro F1 (unweighted average — this is the key metric; weights minority classes equally)
- Weighted F1 (for comparison with prior work)
- Confusion matrix (full 12-class)
- ROC-AUC per class (one-vs-rest)

**Always report Soiling F1 explicitly.** This is the headline result.

**Ablation table structure:**
```
| Config                          | Macro F1 | Soiling F1 | HotSpot F1 |
|---------------------------------|----------|------------|------------|
| Raw + flat softmax (baseline)   |          |            |            |
| Raw + hierarchical              |          |            |            |
| Residual + flat softmax         |          |            |            |
| Residual + hierarchical (ours)  |          |            |            |
| + focal loss                    |          |            |            |
| + thermal MixUp                 |          |            |            |
```

---

## File structure (recommended)

```
pv-fault-classification/
├── data/
│   ├── images/                  # raw 24x40 JPEG images from dataset
│   ├── module_metadata.json     # class labels
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── src/
│   ├── dataset.py               # PyTorch Dataset class
│   ├── estimator.py             # baseline temperature estimator (Stage 1)
│   ├── residual.py              # residual map generator (Stage 2)
│   ├── model.py                 # hierarchical classifier (Stage 3)
│   ├── losses.py                # focal loss, class-weighted loss
│   ├── augmentation.py          # thermal-aware augmentations
│   ├── train.py                 # training loop
│   └── evaluate.py              # full evaluation with per-class metrics
├── notebooks/
│   ├── 01_eda.ipynb             # class distribution, image visualizations
│   ├── 02_baseline_estimator.ipynb
│   ├── 03_residual_analysis.ipynb   # visualize ΔT maps per class
│   └── 04_ablation.ipynb
├── configs/
│   └── default.yaml             # hyperparameters
├── requirements.txt
└── README.md
```

---

## Key implementation details

### Dataset loading
```python
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class InfraredSolarDataset(Dataset):
    CLASS_TO_IDX = {
        'No-Anomaly': 0, 'Cell': 1, 'Cell-Multi': 2,
        'Cracking': 3, 'Hot-Spot': 4, 'Hot-Spot-Multi': 5,
        'Shadowing': 6, 'Diode': 7, 'Diode-Multi': 8,
        'Vegetation': 9, 'Soiling': 10, 'Offline-Module': 11
    }
    
    # Physical grouping for hierarchical classifier
    ENVIRONMENTAL = ['Soiling', 'Shadowing', 'Vegetation']
    ELECTRICAL = ['Cell', 'Cell-Multi', 'Hot-Spot', 'Hot-Spot-Multi',
                  'Diode', 'Diode-Multi', 'Cracking', 'Offline-Module']
```

### Focal loss
```python
import torch.nn.functional as F

def focal_loss(logits, targets, alpha=None, gamma=2.0):
    ce_loss = F.cross_entropy(logits, targets, weight=alpha, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = ((1 - pt) ** gamma) * ce_loss
    return focal.mean()
```

### Thermal MixUp (same-class only)
```python
def thermal_mixup(img_a, img_b, label, lam=None):
    """Only mix images of the same class to preserve physical validity."""
    if lam is None:
        lam = torch.distributions.Beta(0.4, 0.4).sample()
    mixed = lam * img_a + (1 - lam) * img_b
    return mixed, label  # label unchanged — both are same class
```

### Residual generation
```python
def generate_residual(image, estimator, device):
    """
    image: torch.Tensor [1, H, W] normalized to [0,1]
    estimator: trained regression model on No-Anomaly samples
    Returns: residual map, same shape as image
    """
    with torch.no_grad():
        baseline_scalar = estimator(image.unsqueeze(0).to(device))
    baseline_map = baseline_scalar.expand_as(image)
    residual = image - baseline_map
    return residual
```

---

## Dependencies

```
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.0          # EfficientNet, Swin, ViT implementations
numpy>=1.24.0
Pillow>=10.0.0
scikit-learn>=1.3.0  # metrics, stratified splits
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
grad-cam>=1.4.0      # for GradCAM explainability
pyyaml>=6.0          # config files
tqdm>=4.65.0
```

---

## Explainability (required for publication)

Use GradCAM on both your model and a raw-image baseline. For each fault class, show side-by-side:
1. Raw thermal image
2. Baseline model GradCAM activation
3. Your residual map
4. Your model GradCAM activation on residual

The claim to verify: your model attends to physically meaningful regions.
- Cell fault → activation centered on the hot cell(s)
- Diode fault → activation on the activated substring boundary
- Soiling → diffuse activation across the surface, not corners or edges
- Shadowing → sharp boundary region

If GradCAM shows activations on corners or image borders for any class, the model is learning image statistics (compression artifacts, vignetting from the camera) rather than fault signatures. Fix with more aggressive augmentation or explicit border masking.

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

cam = GradCAM(model=classifier, target_layers=[classifier.features[-1]])
grayscale_cam = cam(input_tensor=residual_batch)
```

---

## Publication target

**Primary:** Solar Energy (Elsevier) — accepts CV+PV work, good impact factor  
**Alternative:** IEEE Transactions on Sustainable Energy, Applied Energy  
**Workshop:** NeurIPS/ICLR workshop on AI for Climate or Earth Sciences (same venue as the original dataset paper)

**Contribution framing for abstract:**
> We present the first open-source implementation of IEC 62446-3-compliant irradiance normalization as a preprocessing step for machine learning-based PV fault classification. By estimating the irradiance condition directly from thermal image statistics and computing a physics-derived ΔT residual map, we decouple fault signatures from environmental confounds. Combined with a physically-motivated hierarchical classifier and thermal-aware augmentation, our approach improves soiling F1 from 0.30 to X.XX and overall macro F1 from 0.73 to X.XX on the InfraredSolarModules benchmark, without discarding any fault class.

---

## Known pitfalls

**Pitfall 1 — Evaluating on fault-only vs. full dataset**  
Some papers report accuracy on fault-only 10k samples (no healthy panels). Others include the full 20k. These numbers are NOT comparable. Always state which split you used and report both.

**Pitfall 2 — Data leakage in baseline estimator**  
The baseline temperature estimator must be trained ONLY on No-Anomaly samples. If you accidentally include any fault samples in estimator training, it will learn to suppress fault signatures in the residual, defeating the purpose.

**Pitfall 3 — Image resize artifacts**  
24×40 → 128×128 is a 5.3× upscale. Nearest-neighbor interpolation creates blocking artifacts that the CNN will learn as spurious features. Use bicubic or bilinear. Verify visually before training.

**Pitfall 4 — Class index consistency**  
`module_metadata.json` uses string class names. Make sure your CLASS_TO_IDX mapping is consistent across all splits, the estimator training, and the classifier training. A mismatch here produces silent, catastrophic bugs.

**Pitfall 5 — Thermal MixUp only within class**  
Standard MixUp (mixing across classes) is physically nonsensical for thermal images — a mixture of a Soiling image and a Diode image has no real-world analog. Restrict MixUp to same-class pairs.

**Pitfall 6 — Reporting macro F1 vs. weighted F1**  
Weighted F1 is dominated by Cell (18.8% of faults) and will look good even if Soiling is 0.0. Always report macro F1 as the primary metric. Papers that only report weighted or overall accuracy are hiding poor minority class performance.

---

## Context for this project

Built by Abdul Moiz (final-year EE, GIK Institute, CGPA 3.35). Power electronics background (Hart textbook, DC-DC converters, rectifiers, inverters, PLECS simulation). Target: CV/ML project that sits at the EE-ML intersection for portfolio and publication. Work environment: MacBook Pro, Claude Desktop with `markitdown-mcp`. The project is being developed using Claude Code for implementation assistance. Dataset is fully public, no data collection needed.
