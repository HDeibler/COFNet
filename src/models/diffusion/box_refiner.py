"""
Diffusion Box Refiner for COFNet.

Iteratively refines box proposals through denoising diffusion.
Based on DiffusionDet: https://arxiv.org/abs/2211.09788
"""

import math
from typing import Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class BoxDenoiser(nn.Module):
    """
    Neural network that predicts noise to remove from noisy boxes.

    Architecture:
    - Time embedding
    - Box embedding
    - Cross-attention to image features
    - Self-attention between boxes
    - Noise prediction head
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(feature_dim),
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim),
        )

        # Box embedding (4D -> feature_dim)
        self.box_embed = nn.Sequential(
            nn.Linear(4, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Noise prediction head
        self.noise_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 4),
        )

    def forward(
        self,
        noisy_boxes: torch.Tensor,
        features: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict noise to remove from noisy boxes.

        Args:
            noisy_boxes: [B, N, 4] noisy box coordinates
            features: [B, L, D] image features from CSF
            timesteps: [B] diffusion timesteps

        Returns:
            [B, N, 4] predicted noise
        """
        B, N, _ = noisy_boxes.shape

        # Embed timesteps
        t_embed = self.time_embed(timesteps)  # [B, D]
        t_embed = t_embed.unsqueeze(1).expand(-1, N, -1)  # [B, N, D]

        # Embed boxes
        box_embed = self.box_embed(noisy_boxes)  # [B, N, D]

        # Combine box and time embeddings
        query = box_embed + t_embed  # [B, N, D]

        # Cross-attention to image features
        output = self.decoder(query, features)  # [B, N, D]

        # Predict noise
        noise_pred = self.noise_head(output)  # [B, N, 4]

        return noise_pred


class DiffusionBoxRefiner(nn.Module):
    """
    Diffusion-based box refinement module.

    Training: Add noise to GT boxes, learn to predict and remove it
    Inference: Start from random boxes, iteratively denoise
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_steps: int = 1000,
        num_heads: int = 8,
        num_layers: int = 6,
    ):
        super().__init__()

        self.num_steps = num_steps
        self.feature_dim = feature_dim

        # Noise schedule
        betas = cosine_beta_schedule(num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # Denoiser network
        self.denoiser = BoxDenoiser(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to boxes.

        Args:
            x_0: [B, N, 4] clean boxes
            t: [B] timesteps
            noise: Optional [B, N, 4] noise to add

        Returns:
            [B, N, 4] noisy boxes
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod = cast(torch.Tensor, self.sqrt_alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = cast(torch.Tensor, self.sqrt_one_minus_alphas_cumprod)
        sqrt_alpha = sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t][:, None, None]

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def forward(
        self,
        boxes: torch.Tensor,
        features: torch.Tensor,
        targets: Optional[List[Dict]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Training forward pass.

        Args:
            boxes: [B, N, 4] initial boxes (can be noisy GT)
            features: [B, L, D] image features
            targets: List of dicts with 'boxes' for training

        Returns:
            refined_boxes: [B, N, 4]
            loss: diffusion training loss (if training)
        """
        B, N, _ = boxes.shape
        device = boxes.device

        if self.training and targets is not None:
            # Training: compute diffusion loss
            # Sample random timesteps
            t = torch.randint(0, self.num_steps, (B,), device=device)

            # Add noise to boxes
            noise = torch.randn_like(boxes)
            noisy_boxes = self.q_sample(boxes, t, noise)

            # Flatten features if needed
            if features.dim() == 4:
                features = features.flatten(2).transpose(1, 2)  # [B, H*W, D]

            # Predict noise
            noise_pred = self.denoiser(noisy_boxes, features, t)

            # Simple MSE loss
            loss = F.mse_loss(noise_pred, noise)

            # Return noisy boxes as "refined" during training
            return noisy_boxes, loss
        else:
            # Inference: return boxes as-is (use sample() for actual refinement)
            return boxes, None

    @torch.no_grad()
    def sample(
        self,
        init_boxes: torch.Tensor,
        features: torch.Tensor,
        num_steps: int = 8,
    ) -> torch.Tensor:
        """
        DDIM-style sampling for fast inference.

        Args:
            init_boxes: [B, N, 4] initial noisy boxes
            features: [B, L, D] or [B, D, H, W] image features
            num_steps: Number of denoising steps

        Returns:
            [B, N, 4] refined boxes
        """
        B, N, _ = init_boxes.shape
        device = init_boxes.device

        # Flatten features if spatial
        if features.dim() == 4:
            features = features.flatten(2).transpose(1, 2)

        # Subsample timesteps for DDIM
        step_size = self.num_steps // num_steps
        timesteps = list(range(0, self.num_steps, step_size))[::-1]

        # Start from input boxes (treated as fully noisy)
        x_t = init_boxes

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self.denoiser(x_t, features, t_batch)

            # DDIM update step
            alphas_cumprod = cast(torch.Tensor, self.alphas_cumprod)
            alpha_t = alphas_cumprod[t]
            alpha_prev = alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0)

            # Predict x_0
            x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            # Clamp to valid range
            x_0_pred = x_0_pred.clamp(0, 1)

            # Compute x_{t-1}
            if i < len(timesteps) - 1:
                x_t = torch.sqrt(alpha_prev) * x_0_pred + torch.sqrt(1 - alpha_prev) * noise_pred
            else:
                x_t = x_0_pred

        return x_t
