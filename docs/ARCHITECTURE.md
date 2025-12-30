# COFNet Architecture Specification

## Overview

COFNet (Continuous Object Field Network) is a next-generation object detection architecture designed to address fundamental limitations in current approaches. This document provides detailed technical specifications for implementation.

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              COFNet Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐                                                           │
│   │   Input     │                                                           │
│   │   Image     │                                                           │
│   │  H × W × 3  │                                                           │
│   └──────┬──────┘                                                           │
│          │                                                                   │
│          ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     Mamba-SSM Backbone                               │  │
│   │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐             │  │
│   │  │  Stem   │──▶│ Stage 1 │──▶│ Stage 2 │──▶│ Stage 3 │             │  │
│   │  │  Conv   │   │  Mamba  │   │  Mamba  │   │  Mamba  │             │  │
│   │  └─────────┘   └────┬────┘   └────┬────┘   └────┬────┘             │  │
│   │                     │             │             │                    │  │
│   │                     └──────┬──────┴─────────────┘                    │  │
│   │                            │ Multi-scale features                    │  │
│   └────────────────────────────┼────────────────────────────────────────┘  │
│                                │                                            │
│                                ▼                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                  Continuous Scale Field (CSF)                        │  │
│   │                                                                      │  │
│   │   F(x, y, s) = Σ φᵢ(x, y) · ψᵢ(s)                                   │  │
│   │                                                                      │  │
│   │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │  │
│   │   │   Spatial    │    │    Scale     │    │   Feature    │         │  │
│   │   │  Encoder φ   │ ×  │  Encoder ψ   │ =  │   Output F   │         │  │
│   │   └──────────────┘    └──────────────┘    └──────────────┘         │  │
│   │                                                                      │  │
│   │   Query any scale s ∈ [0, 1] continuously                           │  │
│   └─────────────────────────────────┬───────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                   Diffusion Box Refiner (DBR)                        │  │
│   │                                                                      │  │
│   │   ┌─────┐    ┌─────┐    ┌─────┐           ┌─────┐                  │  │
│   │   │ t=T │───▶│t=T-1│───▶│t=T-2│───...────▶│ t=0 │                  │  │
│   │   │noise│    │     │    │     │           │clean│                  │  │
│   │   └─────┘    └─────┘    └─────┘           └─────┘                  │  │
│   │                                                                      │  │
│   │   b_t = b_{t-1} + ε_θ(b_{t-1}, CSF_features, t)                     │  │
│   │                                                                      │  │
│   │   Training: T=1000 steps (DDPM)                                     │  │
│   │   Inference: T=4-8 steps (DDIM)                                     │  │
│   └─────────────────────────────────┬───────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      Detection Head                                  │  │
│   │                                                                      │  │
│   │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐              │  │
│   │   │   Class     │   │    Box      │   │   Score     │              │  │
│   │   │   Logits    │   │   Coords    │   │  Confidence │              │  │
│   │   └─────────────┘   └─────────────┘   └─────────────┘              │  │
│   └─────────────────────────────────┬───────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│                           Final Detections                                  │
│                        (class, bbox, score) × N                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Mamba-SSM Backbone

#### Purpose
Replace transformer-based backbones (ViT, Swin) with State Space Models for O(n) complexity instead of O(n²).

#### Architecture

```python
class MambaBackbone(nn.Module):
    """
    Mamba-based backbone for object detection.

    Architecture:
    - Stem: Standard conv layers (stride 4 downsampling)
    - Stage 1-4: Mamba blocks with progressive downsampling
    - Output: Multi-scale features at 1/8, 1/16, 1/32 resolution
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: list = [96, 192, 384, 768],
        depths: list = [2, 2, 6, 2],
        d_state: int = 16,  # SSM state dimension
        d_conv: int = 4,    # Local convolution width
        expand: int = 2,    # Block expansion factor
    ):
        ...

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Args:
            x: Input image [B, 3, H, W]

        Returns:
            Dictionary with multi-scale features:
            - 'p3': [B, C1, H/8, W/8]
            - 'p4': [B, C2, H/16, W/16]
            - 'p5': [B, C3, H/32, W/32]
        """
        ...
```

#### Key Design Choices

1. **Selective State Space**: Use selective scan (Mamba-1) or SSD (Mamba-2)
2. **Bidirectional scanning**: Scan both forward and backward for 2D spatial data
3. **Local convolution**: d_conv=4 for local context before SSM
4. **Progressive downsampling**: 2x downsample between stages

#### Comparison with Alternatives

| Backbone | Complexity | Memory (1280²) | Speed |
|----------|------------|----------------|-------|
| ViT-L | O(n²) | ~24GB | 1.0x |
| Swin-L | O(n·w²) | ~16GB | 1.5x |
| **Mamba-L** | O(n) | ~8GB | 2.5x |

---

### 2.2 Continuous Scale Field (CSF)

#### Purpose
Replace discrete FPN levels with a continuous representation that can be queried at any scale.

#### Mathematical Formulation

```
F(x, y, s) = Σᵢ φᵢ(x, y) · ψᵢ(s)

where:
- φᵢ(x, y) : Spatial features from backbone stage i
- ψᵢ(s)    : Scale encoding function
- s ∈ [0, 1] : Continuous scale parameter (0=small, 1=large)
```

#### Scale Encoding Options

**Option A: Fourier Features**
```python
def fourier_scale_encoding(s: Tensor, num_freqs: int = 10) -> Tensor:
    """
    Encode scale using Fourier features.

    Args:
        s: Scale values [B, N] in range [0, 1]
        num_freqs: Number of frequency bands

    Returns:
        Encoded scale [B, N, 2*num_freqs]
    """
    freqs = 2 ** torch.linspace(0, num_freqs-1, num_freqs)
    s_scaled = s[..., None] * freqs * math.pi
    return torch.cat([torch.sin(s_scaled), torch.cos(s_scaled)], dim=-1)
```

**Option B: SIREN (Sinusoidal Representation Networks)**
```python
class SIRENScaleEncoder(nn.Module):
    """
    SIREN-based scale encoding with sinusoidal activations.
    """
    def __init__(self, in_dim: int = 1, hidden_dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            SIRENLayer(in_dim, hidden_dim, is_first=True),
            SIRENLayer(hidden_dim, hidden_dim),
            SIRENLayer(hidden_dim, out_dim),
        )

    def forward(self, s: Tensor) -> Tensor:
        return self.net(s)
```

**Option C: Learnable Scale Embeddings**
```python
class LearnableScaleEncoder(nn.Module):
    """
    Learnable continuous scale embeddings via MLP.
    """
    def __init__(self, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        self.net = nn.Sequential(*layers)

    def forward(self, s: Tensor) -> Tensor:
        return self.net(s.unsqueeze(-1))
```

#### CSF Module Implementation

```python
class ContinuousScaleField(nn.Module):
    """
    Continuous Scale Field for multi-scale feature queries.
    """

    def __init__(
        self,
        in_channels: list = [192, 384, 768],  # From backbone stages
        hidden_dim: int = 256,
        scale_encoding: str = 'fourier',  # 'fourier', 'siren', 'learnable'
    ):
        super().__init__()

        # Project backbone features to common dimension
        self.projections = nn.ModuleList([
            nn.Conv2d(c, hidden_dim, 1) for c in in_channels
        ])

        # Scale encoder
        if scale_encoding == 'fourier':
            self.scale_encoder = FourierScaleEncoder(hidden_dim)
        elif scale_encoding == 'siren':
            self.scale_encoder = SIRENScaleEncoder(1, hidden_dim, hidden_dim)
        else:
            self.scale_encoder = LearnableScaleEncoder(hidden_dim)

        # Feature combiner
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        features: dict[str, Tensor],  # Multi-scale backbone features
        query_scales: Tensor,          # [B, N] scales in [0, 1]
        query_positions: Tensor,       # [B, N, 2] normalized (x, y) positions
    ) -> Tensor:
        """
        Query features at arbitrary scales and positions.

        Returns:
            [B, N, hidden_dim] features for each query
        """
        # Encode query scales
        scale_emb = self.scale_encoder(query_scales)  # [B, N, D]

        # Sample spatial features at query positions (bilinear interpolation)
        # Weighted combination based on scale proximity to each feature level
        spatial_feats = self._sample_multiscale(features, query_positions, query_scales)

        # Combine spatial and scale features
        combined = torch.cat([spatial_feats, scale_emb], dim=-1)
        output = self.combiner(combined)

        return output
```

#### Benefits Over Discrete FPN

| Aspect | Discrete FPN | Continuous Scale Field |
|--------|--------------|------------------------|
| Scale levels | 5 fixed (P3-P7) | Infinite (continuous) |
| Small objects | May fall between levels | Query exact scale |
| Memory | 5x feature maps | 1x + lightweight query |
| Flexibility | Fixed at design time | Adaptive per-object |

---

### 2.3 Diffusion Box Refiner (DBR)

#### Purpose
Replace single-shot box regression with iterative refinement using diffusion models.

#### Core Algorithm

```
Training:
1. Given ground truth boxes b₀
2. Add noise: b_t = √(ᾱ_t) * b₀ + √(1-ᾱ_t) * ε, where ε ~ N(0, I)
3. Predict noise: ε̂ = model(b_t, features, t)
4. Loss: L = ||ε - ε̂||²

Inference:
1. Start with random boxes b_T ~ N(0, I)
2. For t = T, T-1, ..., 1:
   a. ε̂ = model(b_t, features, t)
   b. b_{t-1} = denoise(b_t, ε̂, t)  # DDIM or DDPM step
3. Return refined boxes b₀
```

#### Architecture

```python
class DiffusionBoxRefiner(nn.Module):
    """
    Diffusion-based iterative box refinement.

    Boxes are represented as (cx, cy, w, h) normalized to [0, 1].
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        num_queries: int = 300,
        num_classes: int = 80,
        num_timesteps: int = 1000,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.num_timesteps = num_timesteps

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Box embedding
        self.box_embed = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        # Output heads
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.box_head = nn.Linear(hidden_dim, 4)
        self.noise_head = nn.Linear(hidden_dim, 4)  # Predict noise for boxes

        # Noise schedule
        self.register_buffer('betas', self._cosine_beta_schedule(num_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, 0))

    def forward(
        self,
        csf_features: Tensor,  # [B, H*W, D] from CSF
        boxes: Tensor,         # [B, N, 4] noisy boxes
        timesteps: Tensor,     # [B] or [B, N] timesteps
    ) -> dict[str, Tensor]:
        """
        Predict noise to denoise boxes.
        """
        B, N, _ = boxes.shape

        # Embed inputs
        time_emb = self.time_embed(timesteps)  # [B, D]
        box_emb = self.box_embed(boxes)        # [B, N, D]

        # Add time embedding to box embeddings
        if time_emb.dim() == 2:
            time_emb = time_emb.unsqueeze(1).expand(-1, N, -1)
        queries = box_emb + time_emb

        # Cross-attend to CSF features
        for layer in self.layers:
            queries = layer(queries, csf_features)

        # Predict outputs
        class_logits = self.class_head(queries)  # [B, N, num_classes]
        noise_pred = self.noise_head(queries)     # [B, N, 4]

        return {
            'class_logits': class_logits,
            'noise_pred': noise_pred,
        }

    @torch.no_grad()
    def inference(
        self,
        csf_features: Tensor,
        num_steps: int = 4,  # DDIM steps for fast inference
    ) -> dict[str, Tensor]:
        """
        Generate boxes from noise using DDIM sampling.
        """
        B = csf_features.shape[0]
        device = csf_features.device

        # Start with random boxes
        boxes = torch.randn(B, self.num_queries, 4, device=device)

        # DDIM timesteps (evenly spaced)
        timesteps = torch.linspace(self.num_timesteps-1, 0, num_steps).long()

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict noise
            outputs = self.forward(csf_features, boxes, t_batch)
            noise_pred = outputs['noise_pred']

            # DDIM step
            boxes = self._ddim_step(boxes, noise_pred, t, timesteps[i+1] if i < len(timesteps)-1 else 0)

        # Final forward for class predictions
        outputs = self.forward(csf_features, boxes, torch.zeros(B, device=device).long())
        outputs['boxes'] = boxes.sigmoid()  # Normalize to [0, 1]

        return outputs
```

#### Training Loss

```python
def diffusion_loss(
    model: DiffusionBoxRefiner,
    csf_features: Tensor,
    gt_boxes: Tensor,      # [B, N, 4] ground truth
    gt_classes: Tensor,    # [B, N] class labels
) -> Tensor:
    """
    DDPM training loss for box refinement.
    """
    B, N, _ = gt_boxes.shape
    device = gt_boxes.device

    # Sample random timesteps
    t = torch.randint(0, model.num_timesteps, (B,), device=device)

    # Add noise to boxes
    noise = torch.randn_like(gt_boxes)
    alpha_t = model.alphas_cumprod[t].view(B, 1, 1)
    noisy_boxes = torch.sqrt(alpha_t) * gt_boxes + torch.sqrt(1 - alpha_t) * noise

    # Predict noise
    outputs = model(csf_features, noisy_boxes, t)

    # Noise prediction loss
    noise_loss = F.mse_loss(outputs['noise_pred'], noise)

    # Classification loss (focal loss)
    class_loss = focal_loss(outputs['class_logits'], gt_classes)

    return noise_loss + class_loss
```

---

### 2.4 Self-Supervised Object Saliency (SOS)

#### Purpose
Learn general "objectness" from unlabeled video, reducing dependence on labeled data.

#### Pretraining Losses

```python
class SOSPretraining(nn.Module):
    """
    Self-Supervised Object Saliency pretraining.
    """

    def __init__(self, backbone: MambaBackbone, csf: ContinuousScaleField):
        super().__init__()
        self.backbone = backbone
        self.csf = csf
        self.object_head = nn.Conv2d(256, 1, 1)  # Binary objectness

    def temporal_consistency_loss(
        self,
        frame_t: Tensor,    # [B, 3, H, W]
        frame_t1: Tensor,   # [B, 3, H, W] next frame
        flow_t_to_t1: Tensor,  # [B, 2, H, W] optical flow
    ) -> Tensor:
        """
        Objects should be detected consistently across frames.
        """
        # Get objectness maps
        obj_t = self.get_objectness(frame_t)
        obj_t1 = self.get_objectness(frame_t1)

        # Warp obj_t to t+1 using flow
        obj_t_warped = warp_with_flow(obj_t, flow_t_to_t1)

        # Consistency loss (ignore occluded regions)
        valid_mask = compute_occlusion_mask(flow_t_to_t1)
        loss = F.mse_loss(obj_t_warped * valid_mask, obj_t1 * valid_mask)

        return loss

    def motion_segmentation_loss(
        self,
        frame: Tensor,
        flow: Tensor,  # Optical flow
    ) -> Tensor:
        """
        Moving regions are likely objects.
        """
        # Objectness prediction
        obj_pred = self.get_objectness(frame)

        # Motion magnitude as pseudo-label
        motion_mag = torch.norm(flow, dim=1, keepdim=True)
        motion_mask = (motion_mag > motion_mag.mean()).float()

        # Binary cross-entropy
        loss = F.binary_cross_entropy_with_logits(obj_pred, motion_mask)

        return loss

    def contrastive_crop_loss(
        self,
        image: Tensor,
    ) -> Tensor:
        """
        Crops of the same object should have similar features.
        """
        # Get objectness and sample high-objectness regions
        obj_map = self.get_objectness(image)
        crops = sample_crops_by_objectness(image, obj_map, num_crops=4)

        # Get features for each crop
        features = [self.backbone(crop) for crop in crops]

        # InfoNCE contrastive loss
        loss = info_nce_loss(features)

        return loss

    def get_objectness(self, image: Tensor) -> Tensor:
        features = self.backbone(image)
        csf_out = self.csf.get_base_features(features)
        objectness = self.object_head(csf_out)
        return objectness
```

#### Pretraining Data Sources

For SkyWatch domain:
1. **YouTube sky timelapses** - Night sky, stars, planes
2. **Drone footage** - Aerial views similar to surveillance
3. **Wildlife cameras** - Birds, bats in flight
4. **Meteor shower videos** - Natural meteorite examples

#### Pretraining Pipeline

```python
def pretrain_sos(
    model: COFNet,
    video_dataset: VideoDataset,
    num_epochs: int = 100,
):
    """
    Self-supervised pretraining pipeline.
    """
    optimizer = AdamW(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for batch in video_dataset:
            frame_t, frame_t1, flow = batch

            # Compute losses
            L_temporal = model.temporal_consistency_loss(frame_t, frame_t1, flow)
            L_motion = model.motion_segmentation_loss(frame_t, flow)
            L_contrastive = model.contrastive_crop_loss(frame_t)

            # Total loss
            loss = L_temporal + 0.5 * L_motion + 0.5 * L_contrastive

            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 3. Training Pipeline

### 3.1 Two-Phase Training

```
Phase 1: Self-Supervised Pretraining (SOS)
├── Data: Unlabeled video (10M+ frames)
├── Duration: ~1 week on 8x A100
├── Output: Pretrained backbone + CSF
└── Losses: Temporal + Motion + Contrastive

Phase 2: Supervised Fine-tuning
├── Data: Labeled detection dataset (COCO, then SkyWatch)
├── Duration: ~24 hours on 8x A100
├── Output: Full COFNet model
└── Losses: Diffusion + Classification + Box regression
```

### 3.2 Loss Functions

```python
def total_loss(outputs, targets):
    """
    Combined loss for COFNet training.
    """
    # Diffusion box refinement loss
    L_diffusion = diffusion_loss(outputs['noise_pred'], targets['noise'])

    # Classification loss (focal loss for imbalance)
    L_cls = focal_loss(outputs['class_logits'], targets['labels'],
                       alpha=0.25, gamma=2.0)

    # Box regression loss (GIoU)
    L_box = giou_loss(outputs['boxes'], targets['boxes'])

    # MAL loss (from DEIM - matchability-aware)
    L_mal = mal_loss(outputs, targets)

    return {
        'loss_diffusion': L_diffusion,
        'loss_cls': L_cls,
        'loss_box': L_box * 5.0,
        'loss_mal': L_mal,
        'loss_total': L_diffusion + L_cls + 5.0 * L_box + L_mal,
    }
```

### 3.3 Data Augmentation

```yaml
# Inherited from DEIM with modifications
augmentations:
  - Mosaic:
      probability: 0.5
      output_size: 640
  - RandomPhotometricDistort:
      probability: 0.5
  - RandomZoomOut:
      fill: 0
  - RandomIoUCrop:
      probability: 0.8
  - RandomHorizontalFlip:
      probability: 0.5
  - Resize:
      size: [1280, 1280]

# COFNet-specific: Multi-scale queries during training
  - RandomScaleQuery:
      min_scale: 0.0
      max_scale: 1.0
      num_queries_per_gt: 3
```

---

## 4. Inference

### 4.1 Standard Inference

```python
@torch.no_grad()
def inference(model: COFNet, image: Tensor) -> list[Detection]:
    """
    Standard inference pipeline.
    """
    # 1. Extract backbone features
    features = model.backbone(image)

    # 2. Get CSF representation
    csf_features = model.csf.get_dense_features(features)

    # 3. Diffusion box refinement (4 DDIM steps)
    outputs = model.dbr.inference(csf_features, num_steps=4)

    # 4. Post-process
    boxes = outputs['boxes']  # [B, N, 4]
    scores = outputs['class_logits'].softmax(-1)  # [B, N, C]

    # 5. Filter by confidence
    detections = []
    for b in range(boxes.shape[0]):
        mask = scores[b].max(-1).values > 0.3
        detections.append(Detection(
            boxes=boxes[b][mask],
            scores=scores[b][mask].max(-1).values,
            labels=scores[b][mask].max(-1).indices,
        ))

    return detections
```

### 4.2 Adaptive Inference (Future)

```python
@torch.no_grad()
def adaptive_inference(model: COFNet, image: Tensor) -> list[Detection]:
    """
    Adaptive compute: more refinement for hard regions.
    """
    # 1. Fast coarse pass (2 DDIM steps)
    coarse_outputs = model.inference(image, num_steps=2)

    # 2. Estimate difficulty per detection
    difficulty = model.difficulty_estimator(coarse_outputs)

    # 3. More steps for hard detections
    hard_mask = difficulty > 0.5
    if hard_mask.any():
        # Re-run with more steps for hard cases
        refined = model.dbr.inference(
            csf_features,
            num_steps=8,
            init_boxes=coarse_outputs['boxes'][hard_mask],
        )
        coarse_outputs['boxes'][hard_mask] = refined['boxes']

    return coarse_outputs
```

---

## 5. Model Configurations

### 5.1 COFNet-S (Small)
```yaml
backbone:
  embed_dims: [64, 128, 256, 512]
  depths: [2, 2, 4, 2]
  d_state: 8

csf:
  hidden_dim: 192
  scale_encoding: fourier

dbr:
  hidden_dim: 192
  num_heads: 6
  num_layers: 4
  num_queries: 300

# ~15M parameters
# ~50 FPS on A100
```

### 5.2 COFNet-B (Base)
```yaml
backbone:
  embed_dims: [96, 192, 384, 768]
  depths: [2, 2, 6, 2]
  d_state: 16

csf:
  hidden_dim: 256
  scale_encoding: siren

dbr:
  hidden_dim: 256
  num_heads: 8
  num_layers: 6
  num_queries: 300

# ~35M parameters
# ~30 FPS on A100
```

### 5.3 COFNet-L (Large)
```yaml
backbone:
  embed_dims: [128, 256, 512, 1024]
  depths: [2, 2, 8, 2]
  d_state: 32

csf:
  hidden_dim: 384
  scale_encoding: siren

dbr:
  hidden_dim: 384
  num_heads: 12
  num_layers: 8
  num_queries: 500

# ~80M parameters
# ~15 FPS on A100
```

---

## 6. Expected Performance

### COCO Benchmark (Predictions)

| Model | AP | AP_S | AP_M | AP_L | FPS |
|-------|-----|------|------|------|-----|
| DEIM-L | 59.5 | 42.1 | 63.2 | 74.8 | 28 |
| YOLO26-X | 58.2 | 40.5 | 62.1 | 73.2 | 45 |
| **COFNet-B** | **61.0** | **45.5** | **64.5** | **75.0** | 30 |
| **COFNet-L** | **63.0** | **48.0** | **66.5** | **77.0** | 15 |

### SkyWatch Benchmark (Predictions)

| Model | mAP50 | Plane | Wildlife | Meteorite |
|-------|-------|-------|----------|-----------|
| DEIM-L | 0.65 | 0.75 | 0.72 | 0.48 |
| **COFNet-B** | **0.72** | **0.80** | **0.78** | **0.58** |
| **COFNet-B+SOS** | **0.78** | **0.84** | **0.82** | **0.68** |

Key improvement: +20% on meteorites due to SSL pretraining.

---

## 7. References

1. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
2. Chen, S., et al. (2023). DiffusionDet: Diffusion Model for Object Detection. ICCV.
3. Sitzmann, V., et al. (2020). Implicit Neural Representations with Periodic Activation Functions. NeurIPS.
4. Huang, S., et al. (2024). DEIM: DETR with Improved Matching for Fast Convergence.
5. Liu, Y., et al. (2024). VMamba: Visual State Space Model. arXiv.
