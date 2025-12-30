# COFNet Implementation Plan

**Purpose**: Step-by-step guide to implementing COFNet from scratch.

---

## Overview

COFNet implementation is divided into four phases, each building on the previous:

1. **Phase 1**: Mamba-CSF Backbone - Replace FPN with continuous scale queries
2. **Phase 2**: Diffusion Box Refiner - Iterative box denoising
3. **Phase 3**: Self-Supervised Pretraining - Learn objectness without labels
4. **Phase 4**: Integration & Evaluation - Full pipeline on SkyWatch

---

## Phase 1: Mamba-CSF Backbone

### Goal
Create a backbone that:
- Uses Mamba (SSM) for O(n) global context
- Outputs a Continuous Scale Field instead of discrete FPN levels
- Allows querying features at ANY scale, not just P3-P7

### Step 1.1: Set Up Mamba Backbone

**Dependencies:**
```bash
pip install mamba-ssm  # Official Mamba implementation
pip install causal-conv1d  # Required for Mamba
```

**Files to create:**
- `src/models/backbone/mamba_backbone.py`
- `src/models/backbone/blocks.py`

**Implementation order:**
1. Implement `MambaBlock` (single SSM layer)
2. Implement `MambaStage` (multiple blocks + downsampling)
3. Implement `MambaBackbone` (4 stages, returns multi-scale features)
4. Verify output shapes match expected [B, C, H/4, W/4], [B, C, H/8, W/8], etc.

**Test:**
```python
backbone = MambaBackbone(in_channels=3, dims=[96, 192, 384, 768])
x = torch.randn(2, 3, 640, 640)
features = backbone(x)
# Should return list of 4 feature maps at different scales
```

### Step 1.2: Implement Continuous Scale Field (CSF)

**Files to create:**
- `src/models/csf/scale_encoder.py`
- `src/models/csf/continuous_scale_field.py`

**Implementation order:**
1. Implement Fourier feature encoding for scale values
2. Implement SIREN-style MLP for scale → embedding
3. Implement CSF that fuses backbone features with scale embeddings
4. Verify continuous scale queries return interpolated features

**Key insight:**
CSF replaces discrete FPN levels. Instead of selecting P3/P4/P5, we query `CSF(x, y, s)` where `s ∈ [0, 1]` is continuous.

**Test:**
```python
csf = ContinuousScaleField(backbone_dims=[96, 192, 384, 768], out_dim=256)
backbone_features = [...]  # From MambaBackbone
scales = torch.linspace(0, 1, 100)  # 100 scale queries
features = csf(backbone_features, scales)
# Should return [B, 100, 256] - features at each queried scale
```

### Step 1.3: Benchmark on COCO

Before moving to Phase 2, verify no regression:
- Train CSF backbone + standard detection head on COCO
- Compare to FPN baseline
- Target: Within 1% AP of FPN

---

## Phase 2: Diffusion Box Refiner

### Goal
Replace single-shot box regression with iterative diffusion refinement.

### Step 2.1: Implement Box Noise Model

**Files to create:**
- `src/models/diffusion/noise_schedule.py`
- `src/models/diffusion/box_encoder.py`

**Implementation:**
1. DDPM noise schedule (cosine or linear)
2. Box encoding: (cx, cy, w, h) → normalized [0, 1]
3. Forward diffusion: add noise to GT boxes
4. Time embedding (sinusoidal)

### Step 2.2: Implement Denoising Network

**Files to create:**
- `src/models/diffusion/denoiser.py`
- `src/models/diffusion/box_refiner.py`

**Architecture:**
```
Input: noisy_boxes [B, N, 4], features [B, N, D], timestep t
│
├── Time Embedding (sinusoidal) → [B, D]
├── Box Embedding (MLP) → [B, N, D]
├── Cross-attention: boxes attend to features
├── Self-attention: boxes attend to each other
├── MLP: predict noise ε
│
Output: predicted_noise [B, N, 4]
```

### Step 2.3: Training Loop

**Files to create:**
- `src/training/diffusion_loss.py`
- `src/training/diffusion_trainer.py`

**Training:**
```python
# Forward diffusion
t = random_timesteps(B)
noise = torch.randn_like(gt_boxes)
noisy_boxes = sqrt_alpha[t] * gt_boxes + sqrt_one_minus_alpha[t] * noise

# Predict noise
pred_noise = denoiser(noisy_boxes, features, t)

# Simple MSE loss
loss = F.mse_loss(pred_noise, noise)
```

### Step 2.4: DDIM Inference

**For fast inference (4-8 steps instead of 1000):**
```python
def ddim_sample(model, features, num_steps=8):
    boxes = torch.randn(B, N, 4)  # Start from noise

    for t in reversed(timesteps[:num_steps]):
        pred_noise = model(boxes, features, t)
        boxes = ddim_step(boxes, pred_noise, t)

    return boxes
```

---

## Phase 3: Self-Supervised Pretraining

### Goal
Learn "objectness" from unlabeled video before fine-tuning on labeled data.

### Step 3.1: Collect Unlabeled Data

**Sources for sky/aerial video:**
- YouTube sky timelapses
- Drone footage datasets
- Airport runway cameras
- Astronomical observation streams

**Target:** 10,000+ video clips, no labels needed

### Step 3.2: Implement Temporal Consistency Loss

**Files to create:**
- `src/training/ssl/temporal_loss.py`
- `src/training/ssl/optical_flow.py`

**Implementation:**
```python
def temporal_consistency_loss(model, frame_t, frame_t1, flow):
    # Get detections on both frames
    dets_t = model(frame_t)
    dets_t1 = model(frame_t1)

    # Warp frame_t detections to frame_t1 using optical flow
    warped_dets = warp_boxes(dets_t, flow)

    # Loss: warped detections should match frame_t1 detections
    return hungarian_matching_loss(warped_dets, dets_t1)
```

### Step 3.3: Implement Motion Segmentation Loss

**Files to create:**
- `src/training/ssl/motion_loss.py`

**Implementation:**
```python
def motion_segmentation_loss(model, frame_t, frame_t1):
    # Compute optical flow magnitude
    flow = compute_flow(frame_t, frame_t1)
    motion_mask = (flow.norm(dim=1) > threshold).float()

    # Model should detect moving regions as objects
    objectness = model.predict_objectness(frame_t)

    return F.binary_cross_entropy(objectness, motion_mask)
```

### Step 3.4: Pretraining Pipeline

**Files to create:**
- `src/training/ssl/pretrain.py`

```python
for batch in unlabeled_video_loader:
    frame_t, frame_t1 = batch['frames']
    flow = compute_optical_flow(frame_t, frame_t1)

    loss_temporal = temporal_consistency_loss(model, frame_t, frame_t1, flow)
    loss_motion = motion_segmentation_loss(model, frame_t, frame_t1)

    loss = loss_temporal + 0.5 * loss_motion
    loss.backward()
```

---

## Phase 4: Integration & Evaluation

### Step 4.1: Full Pipeline Integration

**Files to create:**
- `src/models/cofnet.py` - Main model class
- `configs/cofnet_base.yaml` - Default configuration

**Integration:**
```python
class COFNet(nn.Module):
    def __init__(self, config):
        self.backbone = MambaBackbone(...)
        self.csf = ContinuousScaleField(...)
        self.diffusion = DiffusionBoxRefiner(...)
        self.cls_head = ClassificationHead(...)

    def forward(self, images, num_diffusion_steps=8):
        # 1. Extract backbone features
        backbone_feats = self.backbone(images)

        # 2. Build continuous scale field
        csf_feats = self.csf(backbone_feats)

        # 3. Generate initial box proposals (random or learned)
        init_boxes = self.generate_proposals(csf_feats)

        # 4. Refine boxes through diffusion
        refined_boxes = self.diffusion.sample(init_boxes, csf_feats, num_diffusion_steps)

        # 5. Classify refined boxes
        classes = self.cls_head(refined_boxes, csf_feats)

        return refined_boxes, classes
```

### Step 4.2: Fine-tune on SkyWatch

```bash
python scripts/train.py \
    --config configs/cofnet_skywatch.yaml \
    --pretrain checkpoints/cofnet_ssl_pretrained.pth \
    --epochs 100 \
    --batch-size 8
```

### Step 4.3: Benchmark

**Metrics to report:**
- mAP50, mAP75, mAP50:95
- Per-class AP (especially meteorite)
- Inference speed (FPS)
- Memory usage

**Baselines to compare:**
- DEIM-D-FINE-L (59.5% AP on COCO)
- RT-DETR
- YOLOv8

---

## File Structure After Implementation

```
COFNet/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cofnet.py              # Main model
│   │   ├── backbone/
│   │   │   ├── __init__.py
│   │   │   ├── mamba_backbone.py
│   │   │   └── blocks.py
│   │   ├── csf/
│   │   │   ├── __init__.py
│   │   │   ├── scale_encoder.py
│   │   │   └── continuous_scale_field.py
│   │   ├── diffusion/
│   │   │   ├── __init__.py
│   │   │   ├── noise_schedule.py
│   │   │   ├── box_encoder.py
│   │   │   ├── denoiser.py
│   │   │   └── box_refiner.py
│   │   └── heads/
│   │       ├── __init__.py
│   │       └── classification_head.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── coco_dataset.py
│   │   ├── skywatch_dataset.py
│   │   └── video_dataset.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── diffusion_loss.py
│   │   └── ssl/
│   │       ├── __init__.py
│   │       ├── temporal_loss.py
│   │       ├── motion_loss.py
│   │       └── pretrain.py
│   └── utils/
│       ├── __init__.py
│       ├── box_ops.py
│       └── visualization.py
├── configs/
│   ├── cofnet_base.yaml
│   ├── cofnet_small.yaml
│   └── cofnet_skywatch.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── export_onnx.py
└── experiments/
    └── .gitkeep
```

---

## Dependencies

```
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
mamba-ssm>=1.0.0
causal-conv1d>=1.0.0
einops>=0.6.0
timm>=0.9.0
pycocotools>=2.0.0
opencv-python>=4.8.0
albumentations>=1.3.0
wandb>=0.15.0
```

---

## Estimated Compute Requirements

| Phase | Hardware | Time |
|-------|----------|------|
| Phase 1 (Mamba-CSF) | 1x A100 | 2-3 days COCO training |
| Phase 2 (Diffusion) | 1x A100 | 3-4 days COCO training |
| Phase 3 (SSL Pretrain) | 4x A100 | 1 week on 100K video clips |
| Phase 4 (SkyWatch) | 1x A100 | 4-6 hours fine-tuning |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Mamba OOM at high resolution | Use gradient checkpointing, reduce hidden dim |
| Diffusion too slow | DDIM with 4-8 steps, distillation if needed |
| SSL doesn't help | Start with supervised baseline, add SSL incrementally |
| Meteorite class still poor | Focal loss, class-balanced sampling, synthetic augmentation |

---

## Success Criteria

1. **Phase 1 Complete**: CSF backbone matches FPN AP on COCO (±1%)
2. **Phase 2 Complete**: Diffusion improves small object AP by 2%+
3. **Phase 3 Complete**: SSL pretrain improves SkyWatch AP by 3%+
4. **Phase 4 Complete**: Overall SkyWatch mAP50 > 0.6, meteorite AP > 0.3
