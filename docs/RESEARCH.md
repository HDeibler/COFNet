# COFNet Research Notes

**Purpose**: Collected research on 2025 object detection innovations that inform COFNet design.

---

## 1. State-of-the-Art Detectors (2025)

### D-FINE + DEIM (CVPR/ICLR 2025)
**Paper**: Dense Efficient Improved Matching for Object Detection
**Repo**: https://github.com/ShihuaHuang95/DEIM

**Key Innovations:**
- **Dense O2O Matching**: 50% faster convergence than sparse matching
- **MAL (Matchability-Aware Loss)**: Improves rare class handling
- **Fine-grained Distribution Refinement**: Better bbox regression

**Results**: 59.5% AP on COCO (SOTA)

**Relevance to COFNet**: MAL loss concept could be adapted for our diffusion training. Shows importance of matching strategy.

---

### YOLO26 (2025)
**Key Innovations:**
- **STAL (Small-Target-Aware Label Assignment)**: Targets small objects specifically
- **ProgLoss**: Progressive loss scaling during training
- **MuSGD**: Custom optimizer for detection

**Results**: ~58% AP on COCO

**Relevance to COFNet**: STAL approach validates our focus on continuous scales. Small objects need special attention.

---

### RF-DETR (2025)
**Key Innovations:**
- **DINOv2 backbone**: Self-supervised ViT features
- **No NMS needed**: End-to-end detection

**Results**: ~57% AP on COCO

**Relevance to COFNet**: Validates self-supervised pretraining helps detection. Our SSL approach builds on this.

---

### YOLOv12 (2025)
**Key Innovations:**
- **R-ELAN**: Revised efficient layer aggregation
- **Area-based attention**: Attention focused on object regions

**Results**: ~56% AP on COCO

**Relevance to COFNet**: Area-based attention aligns with our adaptive compute ideas.

---

## 2. State Space Models (Mamba)

### Mamba: Linear-Time Sequence Modeling
**Paper**: https://arxiv.org/abs/2312.00752
**Repo**: https://github.com/state-spaces/mamba

**Key Ideas:**
- Selective State Spaces (S6) - input-dependent gating
- O(n) complexity vs O(n²) for attention
- Hardware-aware implementation (parallel scan)

**Why for COFNet:**
- Can process 1280x1280 images without memory explosion
- Global context like transformers but linear cost
- Proven effective for vision (VMamba, Mamba-ND)

---

### Vision Mamba (VMamba)
**Paper**: https://arxiv.org/abs/2401.10166

**Key Ideas:**
- Cross-scan for 2D spatial data (4 directions)
- Hierarchical stages like ConvNets
- Competitive with ViT on ImageNet

**For COFNet**: Use as starting point for backbone design.

---

### Mamba YOLO (AAAI 2025)
**Key Ideas:**
- Mamba backbone for detection
- State Space Path Aggregation Network (SS-PAN)
- Bidirectional scanning

**Results**: Competitive with YOLOv8 at lower compute

**For COFNet**: Validates Mamba works for detection.

---

### MambaNeXt-YOLO
**Key Ideas:**
- Enhanced Mamba blocks with gating
- Multi-scale feature aggregation

**For COFNet**: Design inspiration for our Mamba stages.

---

### Super Mamba (Small Object Detection)
**Key Ideas:**
- Specifically designed for small objects
- Enhanced receptive field handling
- Efficient multi-scale processing

**For COFNet**: Directly relevant to SkyWatch use case.

---

## 3. Diffusion Models for Detection

### DiffusionDet
**Paper**: https://arxiv.org/abs/2211.09788
**Repo**: https://github.com/ShoufaChen/DiffusionDet

**Key Ideas:**
- Treat detection as denoising diffusion
- Start from random boxes, iteratively refine
- No need for anchors or NMS
- Progressive refinement helps hard cases

**Architecture:**
```
Random boxes → Encoder → Cross-attention with features → Predict noise → Denoise → Repeat
```

**Results**: 46.7% AP on COCO (competitive with DETR)

**For COFNet**: Core inspiration for our Diffusion Box Refiner. We adapt their architecture to work with CSF features.

---

### DiffusionEngine
**Key Ideas:**
- Use diffusion to generate synthetic training data
- Domain-specific data synthesis
- Augments limited datasets

**For COFNet**: Could use for meteorite class augmentation (only 54 samples).

---

### ODGEN (Apple)
**Key Ideas:**
- Domain-specific object generation
- Controlled generation conditioned on class/pose
- Synthetic data improves detection

**For COFNet**: Another option for data augmentation.

---

## 4. Self-Supervised Detection

### LodeSTAR: Self-Supervised Single-Shot Detection
**Key Ideas:**
- Learn to detect without labels
- Motion as supervision signal
- Temporal consistency across frames

**For COFNet**: Directly informs our Self-Supervised Object Saliency (SOS) module.

---

### Temporal Consistency Learning
**General Approach:**
1. Get detections on frame t
2. Warp to frame t+1 using optical flow
3. Loss: warped detections should match frame t+1 detections

**Benefits:**
- No labels needed
- Learns object permanence
- Works with any video

---

### Motion Segmentation as Supervision
**Approach:**
1. Compute optical flow between frames
2. High-motion regions = likely objects (in most scenes)
3. Train detector to find motion regions

**Limitations:**
- Doesn't work for static objects
- Confuses with camera motion

**Mitigation**: Use with temporal consistency (static objects persist across frames).

---

## 5. Implicit Neural Representations

### SIREN: Sinusoidal Representation Networks
**Paper**: https://arxiv.org/abs/2006.09661
**Repo**: https://github.com/vsitzmann/siren

**Key Ideas:**
- Use sin(wx + b) activations instead of ReLU
- Better at representing continuous signals
- Can encode arbitrary resolution

**For COFNet**: Used in our Continuous Scale Field. Scale embedding uses SIREN-style encoding.

**Code:**
```python
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        # Special initialization
        with torch.no_grad():
            self.linear.weight.uniform_(-1/in_features, 1/in_features)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
```

---

### Fourier Feature Positional Encoding
**Paper**: NeRF (Mildenhall et al.)

**Key Ideas:**
- Map low-dimensional input to high-dimensional features
- Uses sin/cos at multiple frequencies
- Enables learning high-frequency functions

**For COFNet**: Alternative to SIREN for scale encoding.

**Code:**
```python
def fourier_encoding(x, num_frequencies=10):
    # x: [B, 1] scale value in [0, 1]
    freqs = 2.0 ** torch.arange(num_frequencies)
    x_freq = x * freqs * math.pi
    return torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)
```

---

### NeRF-Det
**Key Ideas:**
- Use NeRF for 3D-aware detection
- Implicit representation of scene

**For COFNet**: Inspiration for continuous representations, though we focus on scale not 3D.

---

## 6. Spiking Neural Networks

### SpikeDet
**Key Ideas:**
- Use spiking neurons instead of continuous activations
- 280x less energy on neuromorphic hardware
- Natural for event cameras

**Results**: Competitive detection at fraction of energy

**For COFNet**: Out of scope for now, but interesting for edge deployment.

---

## 7. Key Papers to Read

### Must Read (Core to COFNet)
1. **Mamba** - https://arxiv.org/abs/2312.00752
2. **DiffusionDet** - https://arxiv.org/abs/2211.09788
3. **SIREN** - https://arxiv.org/abs/2006.09661
4. **D-FINE + DEIM** - https://github.com/ShihuaHuang95/DEIM

### Should Read (Relevant Ideas)
5. **VMamba** - https://arxiv.org/abs/2401.10166
6. **NeRF** - https://arxiv.org/abs/2003.08934 (Fourier features)
7. **DETR** - https://arxiv.org/abs/2005.12872 (Set prediction)
8. **DINO** - https://arxiv.org/abs/2104.14294 (Self-supervised)

### Nice to Have (Extended Context)
9. **LodeSTAR** - Self-supervised detection
10. **DDPM** - https://arxiv.org/abs/2006.11239 (Diffusion fundamentals)
11. **DDIM** - https://arxiv.org/abs/2010.02502 (Fast sampling)

---

## 8. Open Research Questions

### For Continuous Scale Field
1. **SIREN vs Fourier features**: Which encodes scale better?
2. **Number of scale queries**: How many needed for good coverage?
3. **Scale range**: Should [0,1] map to absolute pixels or relative to image?

### For Diffusion Box Refiner
1. **Noise schedule**: Cosine vs linear for boxes?
2. **Number of steps**: Minimum for good detection (target: 4-8)?
3. **Initial proposals**: Random vs learned query embeddings (like DETR)?

### For Self-Supervised Learning
1. **Video data source**: What's the best domain-relevant data?
2. **Loss weighting**: Temporal vs motion vs multi-view?
3. **Pretraining duration**: How much unlabeled data needed?

### For Integration
1. **Training order**: End-to-end or modular?
2. **Loss balancing**: Classification vs box refinement?
3. **Inference optimization**: Can diffusion steps be reduced with distillation?

---

## 9. Relevant Datasets

### For Pretraining
| Dataset | Size | Notes |
|---------|------|-------|
| COCO | 118K train | Standard benchmark |
| Objects365 | 2M images | Large-scale pretraining |
| OpenImages | 9M images | Diverse categories |

### For SSL Pretraining
| Source | Size | Notes |
|--------|------|-------|
| YouTube sky videos | Unlimited | Domain-relevant |
| Airport cameras | Variable | Moving objects in sky |
| Drone footage | Variable | Aerial perspective |

### Target Dataset
| Dataset | Size | Notes |
|---------|------|-------|
| SkyWatch | 8,469 images | 3 classes, imbalanced |

---

## 10. Code References

### Mamba
```python
# Install
pip install mamba-ssm causal-conv1d

# Usage
from mamba_ssm import Mamba
mamba = Mamba(d_model=256, d_state=16, d_conv=4, expand=2)
```

### DiffusionDet
```python
# Clone
git clone https://github.com/ShoufaChen/DiffusionDet

# Key files
# - diffusiondet/detector.py (main model)
# - diffusiondet/head.py (diffusion head)
# - diffusiondet/loss.py (training loss)
```

### SIREN
```python
# Clone
git clone https://github.com/vsitzmann/siren

# Key idea: sin activation with omega_0 scaling
def siren_forward(x, weight, bias, omega_0=30):
    return torch.sin(omega_0 * F.linear(x, weight, bias))
```

---

## 11. Summary: What COFNet Borrows

| Innovation | Source | Our Adaptation |
|------------|--------|----------------|
| State Space Model | Mamba | Backbone for O(n) global context |
| Iterative refinement | DiffusionDet | Box denoising for precision |
| Continuous encoding | SIREN/NeRF | Scale field instead of discrete FPN |
| Self-supervision | LodeSTAR/DINO | Temporal + motion pretraining |
| Dense matching | DEIM | Potential loss adaptation |

**Novel Contributions:**
1. First to combine SSM + Diffusion + Continuous Scale
2. Unified framework for multi-scale detection without FPN
3. Domain-specific SSL for rare class handling (meteorite)
