# COFNet Project Context Document

**Purpose**: This document provides full context for any AI assistant or developer continuing work on this project. Read this FIRST before making any changes.

**Last Updated**: December 30, 2025
**Project Status**: Research & Design Phase

---

## 1. Background: The SkyWatch Project

### What is SkyWatch?
SkyWatch is an object detection project for detecting small objects in night sky imagery:
- **Planes** (~741 annotations) - aircraft in flight
- **WildLife** (~863 annotations) - birds, bats
- **Meteorites** (~54 annotations) - shooting stars, space debris

### Dataset Details
- **Location**: HuggingFace `Deibler/skywatch-dataset`
- **Format**: COCO JSON
- **Images**: 1080x720 RGB
- **Splits**:
  - Train: 4,934 images, 2,799 annotations
  - Valid: 3,028 images, 1,658 annotations
  - Test: 507 images, 283 annotations

### Key Challenge
The meteorite class has only **54 samples** - extreme class imbalance. Objects are also very small in the frame (planes at altitude, distant wildlife, tiny meteorite streaks).

---

## 2. What We Tried: D-FINE + DEIM

### Why D-FINE + DEIM?
We researched 2025 SOTA object detectors and chose DEIM (Dense Efficient Improved Matching) with D-FINE because:
- **59.5% AP on COCO** (state-of-the-art as of early 2025)
- **50% faster convergence** via Dense O2O matching
- **MAL (Matchability-Aware Loss)** helps rare class handling
- **Fine-grained Distribution Refinement** for precise bbox regression

### Implementation Attempt
Created notebook at: `notebooks/experiments/dfine/01_train_dfine_deim.ipynb`

### Issues Encountered

#### 1. Torchvision Compatibility (CRITICAL)
**Problem**: DEIM uses `torchvision.transforms.v2` which changed API in v0.21+
- Transforms like `RandomPhotometricDistort`, `RandomZoomOut`, `RandomIoUCrop` raise `NotImplementedError`
- Root cause: torchvision >= 0.21 requires `transform()` method override, DEIM only has `_transform()`

**Solution**: Pin `torchvision==0.18.1` before cloning DEIM
```python
!pip install torchvision==0.18.1 --quiet
```

**Open PRs in DEIM repo**:
- [PR #47](https://github.com/ShihuaHuang95/DEIM/pull/47) - torchvision >= 0.21 support
- [PR #104](https://github.com/ShihuaHuang95/DEIM/pull/104) - NotImplementedError fix

#### 2. Pretrained Weights
**Problem**: GitHub releases don't exist, weights are on Google Drive
**Solution**: Use gdown
```python
import gdown
GDRIVE_ID = "1PIRf02XkrA2xAD3wEiKE2FaamZgSGTAr"
gdown.download(id=GDRIVE_ID, output="deim_hgnetv2_l_coco.pth")
```

#### 3. Memory (OOM)
**Problem**: Default batch_size=32 with 1280x1280 causes OOM on most GPUs
**Solution**: Set `total_batch_size: 8` in config

#### 4. Dataset Format
DEIM expects:
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── annotations/
    ├── instances_train.json
    └── instances_val.json
```

SkyWatch format:
```
data/processed/
├── train/images/
├── valid/images/
├── train_coco.json
└── valid_coco.json
```

Must copy/rename files to match DEIM expectations.

### Working Config
```yaml
__include__:
  - ./deim_hgnetv2_l_coco.yml

output_dir: ./output/skywatch_dfine_l
num_classes: 3
remap_mscoco_category: False  # CRITICAL for custom datasets
eval_spatial_size: [1280, 1280]

train_dataloader:
  dataset:
    img_folder: ./dataset/images/train
    ann_file: ./dataset/annotations/instances_train.json
  total_batch_size: 8
  collate_fn:
    base_size: 1280

val_dataloader:
  dataset:
    img_folder: ./dataset/images/val
    ann_file: ./dataset/annotations/instances_val.json
  total_batch_size: 8
  collate_fn:
    base_size: 1280
```

---

## 3. Research Conducted

### 2025 Object Detection Landscape

| Model | Key Innovation | AP (COCO) |
|-------|----------------|-----------|
| **YOLO26** | STAL (Small-Target-Aware Label Assignment), ProgLoss, MuSGD optimizer | ~58% |
| **D-FINE + DEIM** | Dense O2O matching, MAL loss, Fine-grained Distribution Refinement | 59.5% |
| **RF-DETR** | DINOv2 backbone, no NMS needed | ~57% |
| **YOLOv12** | R-ELAN, area-based attention | ~56% |

### Emerging Paradigms Researched

1. **State Space Models (Mamba)**
   - Linear O(n) complexity vs O(n²) for transformers
   - MambaNeXt-YOLO, Mamba YOLO (AAAI 2025)
   - Super Mamba for small object detection

2. **Diffusion Models for Detection**
   - DiffusionDet: iterative box refinement
   - DiffusionEngine: synthetic data generation
   - ODGEN (Apple): domain-specific data synthesis

3. **Spiking Neural Networks**
   - SpikeDet: 280x less energy on neuromorphic hardware
   - Natural for event cameras / temporal data

4. **Self-Supervised Learning**
   - Learn "objectness" from unlabeled video
   - Motion, temporal, multi-view consistency
   - LodeSTAR: single-shot self-supervised detection

5. **Implicit Neural Representations**
   - Continuous representations (SIREN, NeRF)
   - NeRF-Det, MonoNeRD for 3D detection

---

## 4. The COFNet Proposal

### Motivation
Current detectors share fundamental limitations:
1. **Discrete scales** - FPN has fixed levels, small objects fall between
2. **Single-shot regression** - no refinement, noisy predictions
3. **Quadratic attention** - memory explodes at high resolution
4. **Fully supervised** - needs massive labeled data
5. **Fixed compute** - same cost for easy/hard regions

### COFNet: Four Core Innovations

#### Innovation 1: Continuous Scale Field (CSF)
Instead of discrete FPN levels (P3, P4, P5...), learn a continuous function:

```
F(x, y, s) = φ(x, y) · ψ(s)
```

- `φ(x,y)` = spatial features from backbone
- `ψ(s)` = scale embedding (Fourier features / SIREN)
- Query ANY scale `s ∈ [0, 1]` continuously
- No more "objects falling between scales"

**Implementation approach**:
- Use SIREN (Sinusoidal Representation Networks) for scale encoding
- Or learnable Fourier feature positional encoding
- Condition on continuous scale instead of discrete level index

#### Innovation 2: Diffusion Box Refiner (DBR)
Replace single-shot regression with iterative refinement:

```
b_t = b_{t-1} + ε_θ(b_{t-1}, F, t)
```

- Start from noisy/random box proposals
- Iteratively denoise over T steps (4-8 for inference)
- Model learns to "grow" accurate boxes
- More steps = more refinement for hard cases

**Implementation approach**:
- Adapt DiffusionDet architecture
- Condition on CSF features instead of FPN
- Train with DDPM loss, inference with DDIM for speed

#### Innovation 3: Mamba-SSM Backbone
Replace transformer attention with State Space Models:

```python
# Pseudocode
x = conv_projection(x)
x = selective_ssm(x)  # Linear O(n) complexity
x = spatial_mamba(x, bidirectional=True)
```

**Benefits**:
- O(n) vs O(n²) complexity
- Can process 1280x1280 without memory explosion
- Global context without attention overhead

**Implementation approach**:
- Use Vision Mamba / VMamba as starting point
- Or adapt Mamba-2 for 2D spatial data
- Replace HGNetv2 backbone in DEIM

#### Innovation 4: Self-Supervised Object Saliency (SOS)
Learn to detect "objects" without labels:

```python
# Temporal consistency (video)
L_temporal = ||Detect(frame_t) - Warp(Detect(frame_{t-1}))||

# Motion segmentation
L_motion = CrossEntropy(ObjectMask, OpticalFlowMask)

# Multi-view consistency
L_multiview = ||F(view_1) - Transform(F(view_2))||
```

**Training pipeline**:
1. Phase 1: Pretrain on unlabeled video (YouTube sky footage, drone footage)
2. Phase 2: Fine-tune on small labeled dataset (SkyWatch)

**Why this helps SkyWatch**:
- Meteorites have only 54 samples
- SSL learns general "objectness" from millions of unlabeled frames
- Domain-relevant pretraining (sky video)

---

## 5. Implementation Plan

### Phase 1: Mamba-CSF Backbone (Week 1-2)
1. Set up Mamba/VMamba as backbone
2. Implement Continuous Scale Field with SIREN
3. Replace FPN with CSF module
4. Benchmark on COCO to verify no regression

### Phase 2: Diffusion Box Refiner (Week 3-4)
1. Integrate DiffusionDet head
2. Modify to work with CSF features
3. Implement DDIM for fast inference
4. Add scale-aware conditioning

### Phase 3: Self-Supervised Pretraining (Week 5-6)
1. Collect unlabeled sky/aerial video
2. Implement temporal consistency loss
3. Implement motion segmentation loss
4. Pretrain on unlabeled data

### Phase 4: Integration & Evaluation (Week 7-8)
1. Combine all components
2. Fine-tune on SkyWatch
3. Benchmark vs DEIM baseline
4. Ablation studies

---

## 6. File Locations

| File | Purpose |
|------|---------|
| `/Users/deibler/Documents/projects/Train/COFNet/` | COFNet project root |
| `/Users/deibler/Documents/projects/Train/notebooks/experiments/dfine/01_train_dfine_deim.ipynb` | DEIM training notebook |
| `/Users/deibler/.claude/plans/virtual-knitting-forest.md` | Original DEIM implementation plan |
| `Deibler/skywatch-dataset` (HuggingFace) | SkyWatch dataset |

---

## 7. Key Decisions Made

1. **High resolution (1280x1280)** - Required for small object detection
2. **COCO pretrained weights** - Better transfer than ImageNet
3. **Batch size 8** - Memory constraint on typical GPUs
4. **torchvision 0.18.1** - Compatibility with DEIM transforms
5. **Minimal config overrides** - Inherit DEIM defaults, only change necessary settings

---

## 8. Open Questions

1. **CSF implementation**: SIREN vs Fourier features vs learnable embeddings?
2. **Mamba variant**: VMamba vs Mamba-2 vs custom spatial adaptation?
3. **Diffusion steps**: How many steps for real-time inference?
4. **SSL data source**: What unlabeled video to use for pretraining?
5. **Adaptive compute**: How to estimate region difficulty?

---

## 9. Commands Reference

### DEIM Training (Colab)
```bash
# After setup and dataset preparation
python train.py \
    -c configs/deim_dfine/deim_dfine_skywatch.yml \
    -t deim_hgnetv2_l_coco.pth \
    --use-amp \
    --seed=42
```

### DEIM Evaluation
```bash
python train.py \
    -c configs/deim_dfine/deim_dfine_skywatch.yml \
    --test-only \
    -r output/skywatch_dfine_l/best.pth
```

---

## 10. Contact & Resources

- **DEIM Repo**: https://github.com/ShihuaHuang95/DEIM
- **Dataset**: https://huggingface.co/datasets/Deibler/skywatch-dataset
- **Mamba**: https://github.com/state-spaces/mamba
- **DiffusionDet**: https://github.com/ShoufaChen/DiffusionDet
- **SIREN**: https://github.com/vsitzmann/siren

---

## 11. For the Next Claude Instance

If you're reading this without prior context:

1. **Primary goal**: Create a novel object detection architecture (COFNet) for small object detection
2. **Secondary goal**: Get DEIM baseline working on SkyWatch dataset
3. **The user**: Experienced ML practitioner, appreciates technical depth, dislikes hand-holding
4. **Code style**: Follow the Philosophy of Inevitability (see CLAUDE.md) - simple, direct, no over-abstraction
5. **Current state**: DEIM notebook should work with torchvision 0.18.1 fix; COFNet is in design phase

**Start by reading**:
1. This document (CONTEXT.md)
2. README.md in COFNet folder
3. The DEIM notebook to understand baseline approach

**Do NOT**:
- Patch/hack the DEIM repo unnecessarily
- Add complexity without clear benefit
- Guess at solutions - research first
- Downgrade packages without verifying it's the right fix
