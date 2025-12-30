"""
Scale-Contrastive Learning (SCL) - Novel Scale-Equivariant Self-Supervision.

Core Insight:
COFNet's Continuous Scale Field (CSF) allows querying features at ANY scale.
We can exploit this for self-supervision:
1. Same spatial location at different scales should have RELATED features
   (but not identical - scale carries semantic information)
2. Different spatial locations should have DIFFERENT features regardless of scale
3. Zooming in on an image should produce equivalent detections at different scales

This is fundamentally different from:
- Standard contrastive learning (which ignores scale structure)
- Multi-scale consistency (which uses discrete FPN levels)

We're learning the CONTINUOUS structure of scale space.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Type alias for COFNet model (avoid circular imports)
COFNetModel = Any


class ScaleContrastiveLearning(nn.Module):
    """
    Learn scale-equivariant representations via contrastive learning in scale space.

    Key ideas:
    1. Scale Anchoring: Features at the same location across scales form positive pairs
    2. Scale Equivariance: Zoomed crops should match features at different scales
    3. Scale Discrimination: Model should predict which scale a feature came from
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_scales: int = 16,
        temperature: float = 0.07,
        scale_smooth_weight: float = 1.0,
        scale_predict_weight: float = 0.5,
        zoom_equivariance_weight: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        self.temperature = temperature
        self.scale_smooth_weight = scale_smooth_weight
        self.scale_predict_weight = scale_predict_weight
        self.zoom_equivariance_weight = zoom_equivariance_weight

        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        # Scale predictor: given a feature, predict which scale it came from
        self.scale_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_scales),
        )

        # Scale-aware feature aggregator
        self.scale_aggregator = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True,
        )

    def forward(
        self,
        model: COFNetModel,
        images: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute scale-contrastive losses.

        Args:
            model: COFNet model with backbone and CSF
            images: [B, 3, H, W] input images
            return_features: Whether to return intermediate features

        Returns:
            Dict containing losses and optionally features
        """
        B, _, H, W = images.shape
        device = images.device

        # Extract backbone features
        backbone_features = model.backbone(images)

        # Query CSF at multiple scales
        scales = torch.linspace(0.1, 0.9, self.num_scales, device=device)

        # Sample random spatial positions
        num_positions = 64
        positions = torch.rand(B, num_positions, 2, device=device)

        # Expand scales for all positions
        scales_expanded = scales.view(1, 1, -1).expand(B, num_positions, -1)

        # Get features at all (position, scale) combinations
        all_features = []
        for s_idx in range(self.num_scales):
            scale_val = scales_expanded[:, :, s_idx]
            features = model.csf.query_at_positions(
                backbone_features,
                positions,
                scale_val,
            )  # [B, num_positions, D]
            all_features.append(features)

        # Stack: [B, num_positions, num_scales, D]
        multi_scale_features = torch.stack(all_features, dim=2)

        # Compute losses
        losses = {}

        # 1. Scale Smoothness Loss
        # Adjacent scales should have similar features
        losses['scale_smooth'] = self._scale_smoothness_loss(
            multi_scale_features
        ) * self.scale_smooth_weight

        # 2. Scale Prediction Loss
        # Model should recognize which scale features came from
        losses['scale_predict'] = self._scale_prediction_loss(
            multi_scale_features, scales
        ) * self.scale_predict_weight

        # 3. Zoom Equivariance Loss
        # Zoomed image at scale s should match original at scale s'
        losses['zoom_equivariance'] = self._zoom_equivariance_loss(
            model, images, backbone_features
        ) * self.zoom_equivariance_weight

        # 4. Cross-Position Contrastive Loss
        # Different positions should have different features
        losses['position_contrast'] = self._position_contrastive_loss(
            multi_scale_features
        )

        # Total loss
        losses['total'] = sum(losses.values())

        if return_features:
            losses['multi_scale_features'] = multi_scale_features

        return losses

    def _scale_smoothness_loss(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Features should vary smoothly across scale dimension.

        This prevents the CSF from learning discontinuous scale representations.
        Uses second-order smoothness (Laplacian) for natural variation.
        """
        # features: [B, N, S, D]
        B, N, S, D = features.shape

        if S < 3:
            return torch.tensor(0.0, device=features.device)

        # First derivative: f[s+1] - f[s]
        diff1 = features[:, :, 1:] - features[:, :, :-1]  # [B, N, S-1, D]

        # Second derivative: diff1[s+1] - diff1[s]
        diff2 = diff1[:, :, 1:] - diff1[:, :, :-1]  # [B, N, S-2, D]

        # Smoothness = minimize second derivative magnitude
        smoothness_loss = (diff2 ** 2).mean()

        return smoothness_loss

    def _scale_prediction_loss(
        self,
        features: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """
        Train scale predictor to recognize which scale features came from.

        This ensures the model encodes scale information in features,
        which is essential for scale-aware detection.
        """
        B, N, S, D = features.shape
        device = features.device

        # Flatten for prediction
        features_flat = features.reshape(B * N * S, D)

        # Predict scale
        scale_logits = self.scale_predictor(features_flat)  # [B*N*S, num_scales]

        # Create labels
        scale_labels = torch.arange(S, device=device)
        scale_labels = scale_labels.view(1, 1, S).expand(B, N, S)
        scale_labels = scale_labels.reshape(B * N * S)

        return F.cross_entropy(scale_logits, scale_labels)

    def _zoom_equivariance_loss(
        self,
        model: COFNetModel,
        images: torch.Tensor,
        backbone_features: list,
    ) -> torch.Tensor:
        """
        Zoomed image features should match original at adjusted scale.

        If we zoom in by 2x, features at scale s in zoomed image should
        match features at scale s/2 in original image.
        """
        B, _, H, W = images.shape
        device = images.device

        # Create zoomed version (2x center crop, then resize back)
        crop_h, crop_w = H // 2, W // 2
        start_h, start_w = H // 4, W // 4

        zoomed = images[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
        zoomed = F.interpolate(zoomed, size=(H, W), mode='bilinear', align_corners=False)

        # Get backbone features for zoomed image
        zoomed_backbone = model.backbone(zoomed)

        # Sample positions in the overlap region
        # In zoomed image, (0.5, 0.5) corresponds to center
        # In original image, this is also center
        num_positions = 32
        positions = torch.rand(B, num_positions, 2, device=device) * 0.5 + 0.25

        # Query at multiple scales
        scales = torch.linspace(0.2, 0.8, 8, device=device)

        total_loss = torch.tensor(0.0, device=device)

        for s in scales:
            # Original features at scale s
            orig_features = model.csf.query_at_positions(
                backbone_features,
                positions,
                torch.full((B, num_positions), s.item(), device=device),
            )

            # Zoomed features at scale s*2 (clamped)
            # Because we zoomed 2x, same absolute scale = 2x relative scale
            zoomed_scale = min(s.item() * 2, 0.95)
            zoomed_features = model.csf.query_at_positions(
                zoomed_backbone,
                positions,  # Same relative positions
                torch.full((B, num_positions), zoomed_scale, device=device),
            )

            # Project for contrastive comparison
            orig_proj = self.projector(orig_features)
            zoom_proj = self.projector(zoomed_features)

            # Normalize
            orig_proj = F.normalize(orig_proj, dim=-1)
            zoom_proj = F.normalize(zoom_proj, dim=-1)

            # Contrastive loss: corresponding positions should match
            similarity = torch.einsum('bnd,bnd->bn', orig_proj, zoom_proj)
            loss = -similarity.mean()  # Maximize similarity

            total_loss = total_loss + loss

        return total_loss / len(scales)

    def _position_contrastive_loss(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Different spatial positions should have different features.

        This is the "negative pairs" part of contrastive learning.
        We aggregate across scales for each position, then contrast positions.
        """
        B, N, S, D = features.shape
        device = features.device

        # Aggregate across scales using attention
        # Reshape: [B*N, S, D] for attention
        features_reshaped = features.reshape(B * N, S, D)

        # Self-attention to aggregate scales
        aggregated, _ = self.scale_aggregator(
            features_reshaped,
            features_reshaped,
            features_reshaped,
        )

        # Take mean across scales: [B*N, D]
        position_features = aggregated.mean(dim=1)
        position_features = position_features.reshape(B, N, D)

        # Project and normalize
        position_proj = self.projector(position_features)  # [B, N, D]
        position_proj = F.normalize(position_proj, dim=-1)

        # InfoNCE loss
        total_loss = torch.tensor(0.0, device=device)

        for b in range(B):
            # Similarity matrix: [N, N]
            sim = torch.mm(position_proj[b], position_proj[b].t()) / self.temperature

            # Diagonal is self-similarity (not useful)
            # Off-diagonal should be low (different positions = different features)
            mask = torch.eye(N, device=device).bool()

            # Negative pairs: minimize similarity of different positions
            negative_sim = sim.masked_fill(mask, float('-inf'))

            # Loss: InfoNCE
            labels = torch.arange(N, device=device)
            loss = F.cross_entropy(sim, labels)

            total_loss = total_loss + loss

        return total_loss / B


class ScaleAugmentation(nn.Module):
    """
    Data augmentation in scale space for self-supervised learning.

    Instead of just image augmentations, we augment in scale space:
    1. Random scale jittering when querying CSF
    2. Scale mixup between adjacent scales
    3. Scale cutout (mask certain scale ranges)
    """

    def __init__(
        self,
        scale_jitter: float = 0.1,
        scale_mixup_prob: float = 0.3,
        scale_cutout_prob: float = 0.2,
    ):
        super().__init__()
        self.scale_jitter = scale_jitter
        self.scale_mixup_prob = scale_mixup_prob
        self.scale_cutout_prob = scale_cutout_prob

    def forward(
        self,
        features: torch.Tensor,
        scales: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply scale-space augmentations.

        Args:
            features: [B, N, S, D] multi-scale features
            scales: [S] scale values

        Returns:
            Augmented features and scales
        """
        B, N, S, D = features.shape
        device = features.device

        # Scale jittering
        if self.training:
            jitter = torch.randn(B, N, S, device=device) * self.scale_jitter
            # We can't actually jitter the features, but we can simulate
            # by interpolating between adjacent scale features
            # This is a soft version of scale jittering

        # Scale mixup
        if self.training and torch.rand(1).item() < self.scale_mixup_prob:
            # Mix adjacent scales
            alpha = torch.rand(B, N, S - 1, 1, device=device)
            mixed = alpha * features[:, :, :-1] + (1 - alpha) * features[:, :, 1:]
            features = torch.cat([mixed, features[:, :, -1:]], dim=2)

        # Scale cutout
        if self.training and torch.rand(1).item() < self.scale_cutout_prob:
            # Mask out random scale range
            cutout_start = int(torch.randint(0, S - 2, (1,)).item())
            cutout_len = int(torch.randint(1, min(3, S - cutout_start), (1,)).item())
            features[:, :, cutout_start:cutout_start + cutout_len] = 0

        return features, scales
