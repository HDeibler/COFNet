# COFNet: Continuous Object Field Network

A radically new approach to object detection combining:
- **Continuous Scale Fields** (implicit neural representations)
- **Generative Box Refinement** (diffusion-based detection)
- **State Space Backbone** (Mamba for linear complexity)
- **Self-Supervised Object Saliency** (learn objectness without labels)

## Why COFNet?

Current object detectors (YOLO, DETR, D-FINE) share fundamental limitations:

| Problem | Current Approach | COFNet Solution |
|---------|------------------|-----------------|
| Discrete scales | Fixed FPN levels (P3-P7) | Continuous scale queries |
| Single-shot prediction | Direct regression | Iterative diffusion refinement |
| Quadratic complexity | O(n²) attention | O(n) state space models |
| Label dependency | Fully supervised | Self-supervised pretraining |
| Fixed compute | Same cost for all regions | Adaptive compute allocation |

## Architecture Overview

```
Input Image
     │
     ▼
┌─────────────────────────────────────────┐
│         Mamba-SSM Backbone              │
│   (Linear complexity global context)    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      Continuous Scale Field (CSF)       │
│   F(x,y,s) = φ(x,y) · ψ(s)             │
│   Query ANY scale continuously          │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│     Diffusion Box Refiner (DBR)         │
│   Iteratively denoise box proposals     │
│   b_t = b_{t-1} + ε_θ(b_{t-1}, F, t)   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│    Self-Supervised Object Saliency      │
│   Temporal + Motion + Multi-view        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
          Final Detections
```

## Project Structure

```
COFNet/
├── README.md                 # This file
├── docs/
│   ├── ARCHITECTURE.md       # Detailed architecture design
│   ├── IMPLEMENTATION.md     # Implementation plan
│   └── RESEARCH.md           # Research notes and references
├── src/
│   ├── models/
│   │   ├── backbone/         # Mamba-SSM backbone
│   │   ├── csf/              # Continuous Scale Field
│   │   ├── diffusion/        # Diffusion Box Refiner
│   │   └── heads/            # Detection heads
│   ├── data/                 # Dataset loaders
│   ├── training/             # Training loops, losses
│   └── utils/                # Utilities
├── configs/                  # YAML configurations
├── experiments/              # Experiment logs
└── scripts/                  # Training/eval scripts
```

## Quick Start

### Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Deibler/COFNet/blob/main/notebooks/train_cofnet_colab.ipynb)

1. Open the notebook in Google Colab
2. Select GPU runtime (Runtime → Change runtime type → GPU)
3. Run all cells - the notebook will:
   - Clone the repository
   - Download the SkyWatch dataset from HuggingFace
   - Train COFNet with self-supervised learning
   - Visualize results

### Local Installation

```bash
# Clone the repository
git clone https://github.com/Deibler/COFNet.git
cd COFNet

# Install dependencies (requires CUDA for Mamba-SSM)
pip install -e .

# Train on SkyWatch dataset
python scripts/train.py --config configs/cofnet_skywatch.yaml
```

### Dataset

The SkyWatch dataset is available on HuggingFace: [`Deibler/skywatch-dataset`](https://huggingface.co/datasets/Deibler/skywatch-dataset)

- **Classes**: Plane, WildLife, meteorite
- **Format**: COCO JSON annotations
- **Splits**: Train (4,934), Valid (3,028), Test (507)

## Roadmap

- [x] Phase 1: Mamba-CSF Backbone
- [x] Phase 2: Diffusion Box Refiner
- [x] Phase 3: Self-Supervised Pretraining (SDSS Framework)
- [ ] Phase 4: Adaptive Compute
- [ ] Phase 5: Benchmark on COCO/SkyWatch

## References

- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [DiffusionDet: Diffusion Model for Object Detection](https://arxiv.org/abs/2211.09788)
- [SIREN: Implicit Neural Representations](https://arxiv.org/abs/2006.09661)
- [D-FINE + DEIM](https://github.com/ShihuaHuang95/DEIM)

## License

MIT License
