"""
Cross-Scale Reconstruction (CSR) - Novel Scale-Space Masked Modeling.

Core Insight:
In COFNet's Continuous Scale Field, features at different scales are related.
We can use this for self-supervision by masking features at some scales
and reconstructing them from other scales.

This is fundamentally different from:
- Masked Image Modeling (MAE, BEiT) which masks spatial patches
- Multi-scale feature matching which uses discrete levels

We're doing MASKED SCALE MODELING - learning the continuous structure
of scale space by reconstruction.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Type alias for COFNet model (avoid circular imports)
COFNetModel = Any


class CrossScaleReconstruction(nn.Module):
    """
    Learn scale relationships by reconstructing masked scale features.

    Key ideas:
    1. Scale Masking: Randomly mask features at certain scales
    2. Cross-Scale Prediction: Predict masked scales from unmasked
    3. Scale Interpolation: Learn smooth transitions between scales
    4. Scale Extrapolation: Predict extreme scales from intermediate
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_scales: int = 16,
        mask_ratio: float = 0.5,
        decoder_dim: int = 128,
        decoder_layers: int = 3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        self.mask_ratio = mask_ratio

        # Scale position embedding
        self.scale_embed = nn.Embedding(num_scales, feature_dim)

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, feature_dim))

        # Cross-scale decoder
        # Uses attention to aggregate information from unmasked scales
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # Scale relationship predictor
        # Given two scale features, predict their relative scale
        self.scale_relation = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),  # Relative scale difference
        )

    def forward(
        self,
        model: COFNetModel,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cross-scale reconstruction losses.

        Args:
            model: COFNet model with backbone and CSF
            images: [B, 3, H, W] input images

        Returns:
            Dict containing losses
        """
        B, _, H, W = images.shape
        device = images.device

        # Extract backbone features
        backbone_features = model.backbone(images)

        # Sample positions
        num_positions = 64
        positions = torch.rand(B, num_positions, 2, device=device)

        # Query CSF at all scales
        scales = torch.linspace(0.05, 0.95, self.num_scales, device=device)

        all_features = []
        for s_idx in range(self.num_scales):
            scale_val = torch.full((B, num_positions), scales[s_idx].item(), device=device)
            features = model.csf.query_at_positions(
                backbone_features,
                positions,
                scale_val,
            )  # [B, num_positions, D]
            all_features.append(features)

        # Stack: [B, num_positions, num_scales, D]
        multi_scale_features = torch.stack(all_features, dim=2)

        losses = {}

        # 1. Masked Scale Reconstruction
        losses['reconstruction'] = self._masked_scale_reconstruction(
            multi_scale_features, scales
        )

        # 2. Scale Interpolation Loss
        losses['interpolation'] = self._scale_interpolation_loss(
            multi_scale_features
        )

        # 3. Scale Extrapolation Loss
        losses['extrapolation'] = self._scale_extrapolation_loss(
            multi_scale_features
        )

        # 4. Scale Ordering Loss
        losses['ordering'] = self._scale_ordering_loss(
            multi_scale_features
        )

        losses['total'] = sum(losses.values())

        return losses

    def _masked_scale_reconstruction(
        self,
        features: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mask random scales and reconstruct them from remaining scales.

        Similar to MAE, but in scale dimension instead of spatial.
        """
        B, N, S, D = features.shape
        device = features.device

        # Create random mask
        num_mask = int(S * self.mask_ratio)
        mask_indices = torch.randperm(S, device=device)[:num_mask]
        keep_indices = torch.randperm(S, device=device)[num_mask:]

        # Separate masked and unmasked features
        masked_features = features[:, :, mask_indices]  # [B, N, num_mask, D]
        keep_features = features[:, :, keep_indices]    # [B, N, S-num_mask, D]

        # Add scale embeddings
        scale_emb = self.scale_embed(keep_indices)  # [S-num_mask, D]
        keep_features = keep_features + scale_emb.unsqueeze(0).unsqueeze(0)

        mask_scale_emb = self.scale_embed(mask_indices)  # [num_mask, D]

        # Create queries for masked positions
        # Use mask token + scale embedding
        queries = self.mask_token.expand(B, N, num_mask, -1) + mask_scale_emb.unsqueeze(0).unsqueeze(0)

        # Flatten for transformer
        keep_flat = keep_features.reshape(B * N, S - num_mask, D)
        query_flat = queries.reshape(B * N, num_mask, D)

        # Decode: predict masked from unmasked
        reconstructed = self.decoder(query_flat, keep_flat)  # [B*N, num_mask, D]
        reconstructed = self.reconstruction_head(reconstructed)

        # Reshape back
        reconstructed = reconstructed.reshape(B, N, num_mask, D)

        # Reconstruction loss
        target = masked_features
        loss = F.mse_loss(reconstructed, target)

        return loss

    def _scale_interpolation_loss(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Features at intermediate scales should be interpolatable
        from neighboring scales.

        f(s) ≈ α * f(s-1) + (1-α) * f(s+1)
        """
        B, N, S, D = features.shape
        device = features.device

        if S < 3:
            return torch.tensor(0.0, device=device)

        # For each middle scale, predict from neighbors
        left = features[:, :, :-2]   # [B, N, S-2, D]
        middle = features[:, :, 1:-1]  # [B, N, S-2, D]
        right = features[:, :, 2:]   # [B, N, S-2, D]

        # Simple linear interpolation should approximate middle
        interpolated = 0.5 * left + 0.5 * right

        # Loss: middle should be close to interpolation
        loss = F.mse_loss(middle, interpolated)

        return loss

    def _scale_extrapolation_loss(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extreme scales should be predictable from intermediate scales.

        This prevents degenerate solutions where extreme scales
        are disconnected from the rest.
        """
        B, N, S, D = features.shape
        device = features.device

        if S < 4:
            return torch.tensor(0.0, device=device)

        # Use middle scales to predict extreme scales
        # Smallest scale
        f0 = features[:, :, 0]
        f1 = features[:, :, 1]
        f2 = features[:, :, 2]

        # Linear extrapolation: f0 ≈ 2*f1 - f2
        extrapolated_min = 2 * f1 - f2

        # Largest scale
        fn = features[:, :, -1]
        fn1 = features[:, :, -2]
        fn2 = features[:, :, -3]

        # Linear extrapolation: fn ≈ 2*fn1 - fn2
        extrapolated_max = 2 * fn1 - fn2

        # Loss
        loss_min = F.mse_loss(f0, extrapolated_min)
        loss_max = F.mse_loss(fn, extrapolated_max)

        return (loss_min + loss_max) / 2

    def _scale_ordering_loss(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scale features should respect ordering - larger scales
        should be distinguishable from smaller scales.

        Uses contrastive learning to enforce scale ordering.
        """
        B, N, S, D = features.shape
        device = features.device

        # Sample pairs of scales
        num_pairs = 32
        idx1 = torch.randint(0, S, (num_pairs,), device=device)
        idx2 = torch.randint(0, S, (num_pairs,), device=device)

        # Ensure different scales
        while (idx1 == idx2).any():
            mask = idx1 == idx2
            num_to_resample = int(mask.sum().item())
            idx2[mask] = torch.randint(0, S, (num_to_resample,), device=device)

        # Get features for scale pairs
        f1 = features[:, :, idx1]  # [B, N, num_pairs, D]
        f2 = features[:, :, idx2]

        # Predict which scale is larger
        combined = torch.cat([f1, f2], dim=-1)  # [B, N, num_pairs, 2D]
        combined_flat = combined.reshape(B * N * num_pairs, 2 * D)

        # Predict relative scale
        pred_diff = self.scale_relation(combined_flat).squeeze(-1)  # [B*N*num_pairs]

        # Ground truth: sign of scale difference
        scale_diff = (idx1 - idx2).float()  # Positive if idx1 > idx2
        scale_diff = scale_diff.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
        scale_diff = scale_diff.reshape(B * N * num_pairs)

        # Binary cross entropy: predict which is larger
        # Use tanh on prediction and scale target to [-1, 1]
        pred_sign = torch.tanh(pred_diff)
        target_sign = torch.sign(scale_diff)

        loss = F.mse_loss(pred_sign, target_sign)

        return loss


class ScaleSpaceAutoencoder(nn.Module):
    """
    Full autoencoder for scale space - encode all scales to latent,
    then decode back to all scales.

    This learns a compact representation of how features vary across scales.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_scales: int = 16,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.num_scales = num_scales

        # Encoder: [num_scales, D] -> [latent_dim]
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim * num_scales, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, latent_dim),
        )

        # Decoder: [latent_dim] -> [num_scales, D]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim * num_scales),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode and decode multi-scale features.

        Args:
            features: [B, N, S, D] multi-scale features

        Returns:
            reconstructed: [B, N, S, D] reconstructed features
            latent: [B, N, latent_dim] latent representation
        """
        B, N, S, D = features.shape

        # Flatten scales
        features_flat = features.reshape(B * N, S * D)

        # Encode
        latent = self.encoder(features_flat)  # [B*N, latent_dim]

        # Decode
        reconstructed_flat = self.decoder(latent)  # [B*N, S*D]

        # Reshape
        reconstructed = reconstructed_flat.reshape(B, N, S, D)
        latent = latent.reshape(B, N, -1)

        return reconstructed, latent

    def reconstruction_loss(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss."""
        reconstructed, _ = self.forward(features)
        return F.mse_loss(reconstructed, features)
