"""
COFNet: Continuous Object Field Network

Main model combining:
- Mamba-SSM Backbone (O(n) global context)
- Continuous Scale Field (query any scale)
- Diffusion Box Refiner (iterative denoising)
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .backbone.mamba_backbone import MambaBackbone
from .csf.continuous_scale_field import ContinuousScaleField
from .diffusion.box_refiner import DiffusionBoxRefiner
from .heads.classification_head import ClassificationHead


class COFNet(nn.Module):
    """
    COFNet: Continuous Object Field Network for small object detection.

    Architecture:
        Image -> MambaBackbone -> ContinuousScaleField -> DiffusionBoxRefiner -> Detections

    Args:
        num_classes: Number of object classes
        backbone_dims: Channel dimensions for each backbone stage
        csf_dim: Output dimension of Continuous Scale Field
        num_queries: Number of object queries (box proposals)
        diffusion_steps_train: Diffusion steps during training
        diffusion_steps_infer: Diffusion steps during inference (DDIM)
    """

    def __init__(
        self,
        num_classes: int = 80,
        backbone_dims: Optional[List[int]] = None,
        csf_dim: int = 256,
        num_queries: int = 300,
        diffusion_steps_train: int = 1000,
        diffusion_steps_infer: int = 8,
    ):
        if backbone_dims is None:
            backbone_dims = [96, 192, 384, 768]
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.diffusion_steps_train = diffusion_steps_train
        self.diffusion_steps_infer = diffusion_steps_infer

        # Mamba-SSM Backbone
        self.backbone = MambaBackbone(
            in_channels=3,
            dims=backbone_dims,
        )

        # Continuous Scale Field
        self.csf = ContinuousScaleField(
            backbone_dims=backbone_dims,
            out_dim=csf_dim,
        )

        # Diffusion Box Refiner
        self.box_refiner = DiffusionBoxRefiner(
            feature_dim=csf_dim,
            num_steps=diffusion_steps_train,
        )

        # Classification Head
        self.cls_head = ClassificationHead(
            feature_dim=csf_dim,
            num_classes=num_classes,
        )

        # Learnable object queries (alternative to random init)
        self.query_embed = nn.Embedding(num_queries, csf_dim)

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Forward pass.

        Args:
            images: [B, 3, H, W] input images
            targets: Optional list of dicts with 'boxes' and 'labels' for training

        Returns:
            dict with 'pred_boxes', 'pred_logits', and optionally 'loss'
        """
        B = images.shape[0]
        device = images.device

        # 1. Extract backbone features (multi-scale)
        backbone_features = self.backbone(images)

        # 2. Build continuous scale field
        csf_features = self.csf(backbone_features)

        # 3. Initialize box proposals
        # During training: can use GT boxes + noise
        # During inference: start from random or learned queries
        if self.training and targets is not None:
            # Use GT boxes with noise for training
            init_boxes = self._prepare_training_boxes(targets, device)
        else:
            # Random boxes for inference
            init_boxes = torch.rand(B, self.num_queries, 4, device=device)

        # 4. Refine boxes through diffusion
        if self.training:
            # Training: compute diffusion loss
            refined_boxes, diffusion_loss = self.box_refiner(
                init_boxes,
                csf_features,
                targets=targets,
            )
        else:
            # Inference: DDIM sampling
            refined_boxes = self.box_refiner.sample(
                init_boxes,
                csf_features,
                num_steps=self.diffusion_steps_infer,
            )
            diffusion_loss = None

        # 5. Classify refined boxes
        # Sample features at predicted box locations
        box_features = self.csf.sample_at_boxes(refined_boxes, backbone_features)
        pred_logits = self.cls_head(box_features)

        outputs = {
            'pred_boxes': refined_boxes,  # [B, num_queries, 4] in (cx, cy, w, h)
            'pred_logits': pred_logits,   # [B, num_queries, num_classes]
        }

        if self.training and diffusion_loss is not None:
            outputs['loss_diffusion'] = diffusion_loss

        return outputs

    def _prepare_training_boxes(
        self,
        targets: List[Dict],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Prepare training boxes from GT with noise added.

        For diffusion training, we start from slightly noisy GT boxes
        and learn to denoise them.
        """
        B = len(targets)
        boxes = torch.zeros(B, self.num_queries, 4, device=device)

        for i, target in enumerate(targets):
            gt_boxes = target['boxes']  # [N, 4]
            num_gt = min(len(gt_boxes), self.num_queries)

            # Copy GT boxes
            boxes[i, :num_gt] = gt_boxes[:num_gt]

            # Fill remaining with random boxes
            if num_gt < self.num_queries:
                boxes[i, num_gt:] = torch.rand(
                    self.num_queries - num_gt, 4, device=device
                )

        return boxes


def build_cofnet(config: dict) -> COFNet:
    """Build COFNet from config dict."""
    return COFNet(
        num_classes=config.get('num_classes', 80),
        backbone_dims=config.get('backbone_dims', [96, 192, 384, 768]),
        csf_dim=config.get('csf_dim', 256),
        num_queries=config.get('num_queries', 300),
        diffusion_steps_train=config.get('diffusion_steps_train', 1000),
        diffusion_steps_infer=config.get('diffusion_steps_infer', 8),
    )
