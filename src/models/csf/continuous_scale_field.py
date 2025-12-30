"""
Continuous Scale Field (CSF) for COFNet.

Replaces discrete FPN levels with continuous scale queries.
Query features at ANY scale, not just P3-P7.
"""

import math
from typing import List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to use torchvision's RoIAlign if available
try:
    from torchvision.ops import roi_align as _roi_align
    HAS_ROI_ALIGN = True
except ImportError:
    _roi_align = None
    HAS_ROI_ALIGN = False


class FourierScaleEncoder(nn.Module):
    """
    Encode continuous scale values using Fourier features.

    Based on NeRF positional encoding.
    """

    def __init__(
        self,
        num_frequencies: int = 10,
        include_input: bool = True,
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        # Precompute frequency bands
        freqs = 2.0 ** torch.arange(num_frequencies)
        self.register_buffer('freqs', freqs)

        self.out_dim = num_frequencies * 2
        if include_input:
            self.out_dim += 1

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: [...] scale values in [0, 1]

        Returns:
            [..., out_dim] Fourier-encoded scale features
        """
        # Expand scale for broadcasting
        s_expanded = s.unsqueeze(-1)  # [..., 1]

        # Compute sin and cos at each frequency
        freqs = cast(torch.Tensor, self.freqs)
        x_freq = s_expanded * freqs * math.pi  # [..., num_freq]
        sin_features = torch.sin(x_freq)
        cos_features = torch.cos(x_freq)

        features = torch.cat([sin_features, cos_features], dim=-1)

        if self.include_input:
            features = torch.cat([s_expanded, features], dim=-1)

        return features


class SirenScaleEncoder(nn.Module):
    """
    Encode continuous scale using SIREN (Sinusoidal Representation Networks).

    Better for representing continuous signals.
    """

    def __init__(
        self,
        out_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 3,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0

        layers = []
        in_dim = 1

        for i in range(num_layers):
            is_first = (i == 0)
            is_last = (i == num_layers - 1)

            layer_out = out_dim if is_last else hidden_dim
            layer = nn.Linear(in_dim, layer_out)

            # SIREN initialization
            with torch.no_grad():
                if is_first:
                    layer.weight.uniform_(-1 / in_dim, 1 / in_dim)
                else:
                    bound = math.sqrt(6 / in_dim) / omega_0
                    layer.weight.uniform_(-bound, bound)

            layers.append(layer)
            in_dim = layer_out

        self.layers = nn.ModuleList(layers)
        self.out_dim = out_dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: [...] scale values in [0, 1]

        Returns:
            [..., out_dim] SIREN-encoded scale features
        """
        x = s.unsqueeze(-1)  # [..., 1]

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.sin(self.omega_0 * x)

        return x


class RoIAlignCustom(nn.Module):
    """
    Custom RoI Align implementation for extracting features from boxes.

    Uses bilinear sampling at multiple points within each box.
    """

    def __init__(
        self,
        output_size: int = 7,
        sampling_ratio: int = 2,
    ):
        super().__init__()
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio

    def forward(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract features from boxes.

        Args:
            features: [B, C, H, W] feature map
            boxes: [B, N, 4] boxes in (cx, cy, w, h) normalized format

        Returns:
            [B, N, C, output_size, output_size] pooled features
        """
        B, C, H, W = features.shape
        _, N, _ = boxes.shape
        device = features.device

        # Convert (cx, cy, w, h) to (x1, y1, x2, y2) in pixel coords
        cx, cy, w, h = boxes.unbind(dim=-1)
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H

        # Create sampling grid for each box
        # Grid points within each box
        output_size = self.output_size
        pooled = torch.zeros(B, N, C, output_size, output_size, device=device)

        for b in range(B):
            for n in range(N):
                # Box coordinates
                bx1, by1, bx2, by2 = x1[b, n], y1[b, n], x2[b, n], y2[b, n]

                # Skip invalid boxes
                if bx2 <= bx1 or by2 <= by1:
                    continue

                # Create grid for this box
                # Sample at cell centers
                step_x = (bx2 - bx1) / output_size
                step_y = (by2 - by1) / output_size

                for i in range(output_size):
                    for j in range(output_size):
                        # Sample point (center of grid cell)
                        px = bx1 + (j + 0.5) * step_x
                        py = by1 + (i + 0.5) * step_y

                        # Normalize to [-1, 1] for grid_sample
                        grid_x = (px / W) * 2 - 1
                        grid_y = (py / H) * 2 - 1

                        # Clamp to valid range
                        grid_x = grid_x.clamp(-1, 1)
                        grid_y = grid_y.clamp(-1, 1)

                        # Sample
                        grid = torch.tensor([[[[grid_x, grid_y]]]], device=device)
                        sampled = F.grid_sample(
                            features[b:b+1],
                            grid,
                            mode='bilinear',
                            padding_mode='zeros',
                            align_corners=False,
                        )
                        pooled[b, n, :, i, j] = sampled.squeeze()

        return pooled


def roi_align_boxes(
    features: torch.Tensor,
    boxes: torch.Tensor,
    output_size: int = 7,
) -> torch.Tensor:
    """
    RoI Align for extracting features from boxes.

    Uses torchvision's optimized implementation if available,
    falls back to custom implementation otherwise.

    Args:
        features: [B, C, H, W] feature map
        boxes: [B, N, 4] boxes in (cx, cy, w, h) normalized format
        output_size: Output spatial size

    Returns:
        [B, N, C, output_size, output_size] pooled features
    """
    B, C, H, W = features.shape
    _, N, _ = boxes.shape
    device = features.device

    # Convert (cx, cy, w, h) normalized to (x1, y1, x2, y2) pixel coords
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H

    if HAS_ROI_ALIGN and _roi_align is not None:
        # Use torchvision's optimized ROI align
        # Format: [batch_idx, x1, y1, x2, y2]
        batch_indices = torch.arange(B, device=device).view(-1, 1).expand(-1, N)
        rois = torch.stack([
            batch_indices.flatten().float(),
            x1.flatten(),
            y1.flatten(),
            x2.flatten(),
            y2.flatten(),
        ], dim=1)  # [B*N, 5]

        # ROI align
        pooled = _roi_align(
            features,
            rois,
            output_size=output_size,
            spatial_scale=1.0,
            sampling_ratio=2,
        )  # [B*N, C, output_size, output_size]

        pooled = pooled.view(B, N, C, output_size, output_size)
    else:
        # Fallback to vectorized grid sampling
        pooled = _roi_align_grid_sample(features, boxes, output_size)

    return pooled


def _roi_align_grid_sample(
    features: torch.Tensor,
    boxes: torch.Tensor,
    output_size: int = 7,
) -> torch.Tensor:
    """
    Vectorized RoI align using grid_sample.
    """
    B, C, H, W = features.shape
    _, N, _ = boxes.shape
    device = features.device
    dtype = features.dtype

    # Convert (cx, cy, w, h) to normalized box coords
    cx, cy, w, h = boxes.unbind(dim=-1)  # Each [B, N]

    # Create sampling grid for output_size x output_size
    # Grid values are offsets from -0.5 to 0.5
    lin = torch.linspace(-0.5 + 0.5/output_size, 0.5 - 0.5/output_size, output_size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(lin, lin, indexing='ij')  # [output_size, output_size]

    # Expand for batch and num_boxes
    grid_x = grid_x.view(1, 1, output_size, output_size)  # [1, 1, S, S]
    grid_y = grid_y.view(1, 1, output_size, output_size)

    # Scale by box size and translate to box center
    # Final coords in normalized [0, 1] image space
    cx = cx.view(B, N, 1, 1)  # [B, N, 1, 1]
    cy = cy.view(B, N, 1, 1)
    w = w.view(B, N, 1, 1)
    h = h.view(B, N, 1, 1)

    sample_x = cx + grid_x * w  # [B, N, S, S]
    sample_y = cy + grid_y * h

    # Convert to grid_sample format [-1, 1]
    sample_x = sample_x * 2 - 1
    sample_y = sample_y * 2 - 1

    # Stack into grid
    grid = torch.stack([sample_x, sample_y], dim=-1)  # [B, N, S, S, 2]

    # Sample features for each box
    # We need to process batch-by-batch since grid_sample expects [B, H, W, 2]
    pooled = []
    for b in range(B):
        # grid[b] is [N, S, S, 2]
        sampled = F.grid_sample(
            features[b:b+1].expand(N, -1, -1, -1),  # [N, C, H, W]
            grid[b],  # [N, S, S, 2]
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )  # [N, C, S, S]
        pooled.append(sampled)

    pooled = torch.stack(pooled, dim=0)  # [B, N, C, S, S]

    return pooled


class ContinuousScaleField(nn.Module):
    """
    Continuous Scale Field: query features at any scale.

    F(x, y, s) = fusion(backbone_features, scale_embedding)

    Instead of discrete FPN levels, we:
    1. Encode scale s as a continuous embedding
    2. Interpolate backbone features at the queried scale
    3. Fuse spatial and scale information
    """

    def __init__(
        self,
        backbone_dims: Optional[List[int]] = None,
        out_dim: int = 256,
        scale_encoder: str = 'fourier',  # 'fourier' or 'siren'
        num_scale_frequencies: int = 10,
        roi_output_size: int = 7,
    ):
        super().__init__()
        if backbone_dims is None:
            backbone_dims = [96, 192, 384, 768]

        self.backbone_dims = backbone_dims
        self.out_dim = out_dim
        self.num_levels = len(backbone_dims)
        self.roi_output_size = roi_output_size

        # Scale encoder
        if scale_encoder == 'fourier':
            self.scale_encoder = FourierScaleEncoder(
                num_frequencies=num_scale_frequencies
            )
            scale_dim = self.scale_encoder.out_dim
        else:
            self.scale_encoder = SirenScaleEncoder(out_dim=64)
            scale_dim = 64

        self.scale_dim = scale_dim

        # Project each backbone level to common dimension
        self.level_projs = nn.ModuleList([
            nn.Conv2d(dim, out_dim, 1)
            for dim in backbone_dims
        ])

        # Learnable scale-level mapping
        # Maps continuous scale [0, 1] to level weights
        self.scale_to_level = nn.Sequential(
            nn.Linear(scale_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(backbone_dims)),
            nn.Softmax(dim=-1),
        )

        # Fuse scale embedding with spatial features
        self.scale_fusion = nn.Sequential(
            nn.Linear(out_dim + scale_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        # Position encoding for spatial queries
        self.pos_encoder = FourierScaleEncoder(num_frequencies=6)
        pos_dim = self.pos_encoder.out_dim * 2  # x and y

        # Spatial query fusion
        self.spatial_fusion = nn.Sequential(
            nn.Linear(out_dim + pos_dim + scale_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        # RoI feature aggregation
        self.roi_pool_proj = nn.Sequential(
            nn.Linear(out_dim * roi_output_size * roi_output_size, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        backbone_features: List[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Build unified feature map from multi-scale backbone features.

        Args:
            backbone_features: List of [B, C_i, H_i, W_i] from backbone
            target_size: Optional (H, W) for output size

        Returns:
            [B, out_dim, H, W] unified feature map
        """
        if target_size is None:
            # Default to largest backbone resolution
            shape = backbone_features[0].shape
            target_size = (shape[2], shape[3])

        B = backbone_features[0].shape[0]
        H, W = target_size[0], target_size[1]

        # Project and upsample all levels to target size
        projected = []
        for i, (feat, proj) in enumerate(zip(backbone_features, self.level_projs)):
            feat_proj = proj(feat)  # [B, out_dim, H_i, W_i]
            feat_up = F.interpolate(
                feat_proj,
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )
            projected.append(feat_up)

        # Average all levels (simple fusion)
        unified = torch.stack(projected, dim=0).mean(dim=0)  # [B, out_dim, H, W]

        return unified

    def query_at_positions(
        self,
        backbone_features: List[torch.Tensor],
        positions: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """
        Query features at specific (x, y, scale) positions.

        Args:
            backbone_features: List of [B, C_i, H_i, W_i] from backbone
            positions: [B, N, 2] normalized (x, y) positions in [0, 1]
            scales: [B, N] scale values in [0, 1]

        Returns:
            [B, N, out_dim] features at queried positions and scales
        """
        B, N, _ = positions.shape
        device = positions.device

        # Encode scales
        scale_embed = self.scale_encoder(scales)  # [B, N, scale_dim]

        # Get level weights from scale
        level_weights = self.scale_to_level(scale_embed)  # [B, N, num_levels]

        # Sample from each level at the specified positions
        sampled_features = []
        for i, (feat, proj) in enumerate(zip(backbone_features, self.level_projs)):
            feat_proj = proj(feat)  # [B, out_dim, H_i, W_i]

            # Convert positions to grid_sample format [-1, 1]
            grid = positions * 2 - 1  # [B, N, 2]
            grid = grid.unsqueeze(2)  # [B, N, 1, 2]

            sampled = F.grid_sample(
                feat_proj,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False,
            )  # [B, out_dim, N, 1]

            sampled = sampled.squeeze(-1).transpose(1, 2)  # [B, N, out_dim]
            sampled_features.append(sampled)

        # Stack and weight by level
        stacked = torch.stack(sampled_features, dim=-1)  # [B, N, out_dim, num_levels]
        weighted = (stacked * level_weights.unsqueeze(2)).sum(dim=-1)  # [B, N, out_dim]

        # Encode positions
        pos_x = self.pos_encoder(positions[..., 0])  # [B, N, pos_dim/2]
        pos_y = self.pos_encoder(positions[..., 1])  # [B, N, pos_dim/2]
        pos_embed = torch.cat([pos_x, pos_y], dim=-1)  # [B, N, pos_dim]

        # Fuse position, scale, and features
        fused = torch.cat([weighted, pos_embed, scale_embed], dim=-1)
        output = self.spatial_fusion(fused)

        return output

    def query_at_scale(
        self,
        backbone_features: List[torch.Tensor],
        scales: torch.Tensor,
        num_samples: int = 49,
    ) -> torch.Tensor:
        """
        Query features at specific continuous scale values with spatial sampling.

        Args:
            backbone_features: List of [B, C_i, H_i, W_i] from backbone
            scales: [B, N] scale values in [0, 1]
            num_samples: Number of spatial samples per scale (sqrt should be int)

        Returns:
            [B, N, out_dim] features at queried scales
        """
        B, N = scales.shape
        device = scales.device
        grid_size = int(math.sqrt(num_samples))

        # Create uniform spatial sampling grid
        lin = torch.linspace(0.1, 0.9, grid_size, device=device)
        grid_y, grid_x = torch.meshgrid(lin, lin, indexing='ij')
        positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [num_samples, 2]

        # Expand for batch and num_scales
        positions = positions.view(1, 1, num_samples, 2).expand(B, N, -1, -1)  # [B, N, num_samples, 2]
        positions = positions.reshape(B, N * num_samples, 2)

        # Expand scales to match positions
        scales_expanded = scales.unsqueeze(-1).expand(-1, -1, num_samples)  # [B, N, num_samples]
        scales_expanded = scales_expanded.reshape(B, N * num_samples)

        # Query at all positions
        features = self.query_at_positions(backbone_features, positions, scales_expanded)  # [B, N*num_samples, out_dim]

        # Reshape and aggregate
        features = features.view(B, N, num_samples, -1)  # [B, N, num_samples, out_dim]
        features = features.mean(dim=2)  # [B, N, out_dim]

        return features

    def sample_at_boxes(
        self,
        boxes: torch.Tensor,
        backbone_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Sample features at box locations using RoI Align.

        Args:
            boxes: [B, N, 4] boxes in (cx, cy, w, h) normalized format
            backbone_features: List of [B, C_i, H_i, W_i] from backbone

        Returns:
            [B, N, out_dim] features at each box location
        """
        B, N, _ = boxes.shape

        # Get unified features
        unified = self.forward(backbone_features)  # [B, out_dim, H, W]
        _, C, H, W = unified.shape

        # RoI Align
        pooled = roi_align_boxes(
            unified,
            boxes,
            output_size=self.roi_output_size,
        )  # [B, N, C, roi_size, roi_size]

        # Flatten and project
        pooled_flat = pooled.view(B, N, -1)  # [B, N, C * roi_size * roi_size]
        box_features = self.roi_pool_proj(pooled_flat)  # [B, N, out_dim]

        # Add scale information based on box size
        cx, cy, w, h = boxes.unbind(dim=-1)
        box_scale = (w * h).sqrt().clamp(min=1e-6)  # Approximate scale from box area
        scale_embed = self.scale_encoder(box_scale)  # [B, N, scale_dim]

        # Fuse with scale
        fused = torch.cat([box_features, scale_embed], dim=-1)
        output = self.scale_fusion(fused)

        return output
