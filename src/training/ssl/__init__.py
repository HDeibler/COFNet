"""
Scale-Diffusion Self-Supervision (SDSS) for COFNet.

A novel self-supervised learning framework that exploits COFNet's unique architecture:
1. Diffusion Convergence Discovery (DCD) - Use diffusion dynamics to discover objects
2. Scale-Contrastive Learning (SCL) - Learn scale-equivariant representations
3. Temporal-Diffusion Consistency (TDC) - Enforce consistency across video frames
4. Cross-Scale Reconstruction (CSR) - Predict masked scale features

This is fundamentally different from existing SSL methods because it uses the
detection architecture itself (diffusion + continuous scale) as the supervision signal.
"""

from .diffusion_convergence import DiffusionConvergenceDiscovery, DiffusionConvergenceLoss
from .scale_contrastive import ScaleContrastiveLearning, ScaleAugmentation
from .temporal_diffusion import TemporalDiffusionConsistency, LightweightFlowNet
from .cross_scale_reconstruction import CrossScaleReconstruction, ScaleSpaceAutoencoder
from .sdss_pretrainer import SDSSPretrainer, SDSSLoss

__all__ = [
    'DiffusionConvergenceDiscovery',
    'DiffusionConvergenceLoss',
    'ScaleContrastiveLearning',
    'ScaleAugmentation',
    'TemporalDiffusionConsistency',
    'LightweightFlowNet',
    'CrossScaleReconstruction',
    'ScaleSpaceAutoencoder',
    'SDSSPretrainer',
    'SDSSLoss',
]
