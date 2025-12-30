"""
Mamba-SSM Backbone for COFNet.

Uses State Space Models for O(n) global context instead of O(n²) attention.
Implements selective state spaces with bidirectional scanning for 2D images.
"""

import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Try to import optimized Mamba, fall back to pure PyTorch
try:
    from mamba_ssm import Mamba as OptimizedMamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model - the core Mamba operation.

    Implements: y = SSM(A, B, C, D)(x)
    Where A, B, C are input-dependent (selective).

    State equation:
        h_t = A_t * h_{t-1} + B_t * x_t
        y_t = C_t * h_t + D * x_t
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[str, int] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # dt_rank controls the bottleneck for dt projection
        self.dt_rank: int = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)

        # Input projection: x -> (z, x, B, C, dt)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # 1D convolution for local context before SSM
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
        )

        # SSM parameters
        # A is structured as a diagonal matrix, initialized with HiPPO
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Input-dependent projections for B, C, dt
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt projection
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # dt bias initialization to cover [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input sequence

        Returns:
            [B, L, D] output sequence
        """
        batch, seqlen, dim = x.shape

        # Input projection
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x, z = xz.chunk(2, dim=-1)  # Each [B, L, d_inner]

        # 1D conv (needs channel-first)
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :seqlen]  # Causal: take first seqlen
        x = rearrange(x, "b d l -> b l d")
        x = F.silu(x)

        # Compute SSM parameters
        x_dbl = self.x_proj(x)  # [B, L, dt_rank + 2*d_state]
        dt, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        dt = self.dt_proj(dt)  # [B, L, d_inner]
        dt = F.softplus(dt)  # Ensure positive

        # Get A (negative for stability)
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        # Run selective scan
        y = self.selective_scan(x, dt, A, B, C, self.D)

        # Gate and output
        y = y * F.silu(z)
        output = self.out_proj(y)

        return output

    def selective_scan(
        self,
        x: torch.Tensor,  # [B, L, D]
        dt: torch.Tensor,  # [B, L, D]
        A: torch.Tensor,  # [D, N]
        B: torch.Tensor,  # [B, L, N]
        C: torch.Tensor,  # [B, L, N]
        D: torch.Tensor,  # [D]
    ) -> torch.Tensor:
        """
        Selective scan operation - the core of Mamba.

        For each position t:
            h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t
            y_t = C_t @ h_t + D * x_t
        """
        batch, seqlen, d_inner = x.shape
        d_state = A.shape[1]
        device = x.device
        dtype = x.dtype

        # Discretize A and B
        # dA = exp(dt * A)
        # dB = dt * B
        dt = dt.unsqueeze(-1)  # [B, L, D, 1]
        A = A.unsqueeze(0).unsqueeze(0)  # [1, 1, D, N]
        dA = torch.exp(dt * A)  # [B, L, D, N]

        B = B.unsqueeze(2)  # [B, L, 1, N]
        dB = dt * B  # [B, L, D, N]

        # Sequential scan (can be parallelized with associative scan)
        h = torch.zeros(batch, d_inner, d_state, device=device, dtype=dtype)
        ys = []

        for t in range(seqlen):
            h = dA[:, t] * h + dB[:, t] * x[:, t:t+1, :].transpose(1, 2)  # [B, D, N]
            y_t = torch.einsum("bdn,bln->bdl", h, C[:, t:t+1])  # [B, D, 1]
            ys.append(y_t.squeeze(-1))  # [B, D]

        y = torch.stack(ys, dim=1)  # [B, L, D]
        y = y + D * x  # Skip connection

        return y


class BidirectionalMamba(nn.Module):
    """
    Bidirectional Mamba for 2D feature maps.

    Processes sequences in both forward and backward directions,
    then combines the results.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.forward_ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.backward_ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.merge = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input sequence

        Returns:
            [B, L, D] output with bidirectional context
        """
        # Forward direction
        y_fwd = self.forward_ssm(x)

        # Backward direction
        x_rev = torch.flip(x, dims=[1])
        y_bwd = self.backward_ssm(x_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])

        # Merge
        y = self.merge(torch.cat([y_fwd, y_bwd], dim=-1))

        return y


class CrossScanMamba(nn.Module):
    """
    Cross-Scan Mamba for 2D images (like VMamba).

    Scans the 2D feature map in 4 directions:
    - Left to right
    - Right to left
    - Top to bottom
    - Bottom to top

    This gives each pixel global context from all directions.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        # Four SSMs for four scan directions
        self.ssm_lr = SelectiveSSM(d_model, d_state, d_conv, expand)  # Left-Right
        self.ssm_rl = SelectiveSSM(d_model, d_state, d_conv, expand)  # Right-Left
        self.ssm_tb = SelectiveSSM(d_model, d_state, d_conv, expand)  # Top-Bottom
        self.ssm_bt = SelectiveSSM(d_model, d_state, d_conv, expand)  # Bottom-Top

        # Merge four directions
        self.merge = nn.Linear(d_model * 4, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, D] flattened 2D features
            H, W: spatial dimensions

        Returns:
            [B, H*W, D] output with 2D global context
        """
        B, L, D = x.shape

        # Reshape to 2D for direction-aware scanning
        x_2d = x.view(B, H, W, D)

        # Left-Right scan (row-wise)
        x_lr = x_2d.reshape(B * H, W, D)
        y_lr = self.ssm_lr(x_lr).view(B, H, W, D)

        # Right-Left scan
        x_rl = torch.flip(x_2d, dims=[2]).reshape(B * H, W, D)
        y_rl = self.ssm_rl(x_rl).view(B, H, W, D)
        y_rl = torch.flip(y_rl, dims=[2])

        # Top-Bottom scan (column-wise)
        x_tb = x_2d.permute(0, 2, 1, 3).reshape(B * W, H, D)
        y_tb = self.ssm_tb(x_tb).view(B, W, H, D).permute(0, 2, 1, 3)

        # Bottom-Top scan
        x_bt = torch.flip(x_2d.permute(0, 2, 1, 3), dims=[1]).reshape(B * W, H, D)
        y_bt = self.ssm_bt(x_bt).view(B, W, H, D)
        y_bt = torch.flip(y_bt, dims=[1]).permute(0, 2, 1, 3)

        # Merge all directions
        y = torch.cat([
            y_lr.reshape(B, L, D),
            y_rl.reshape(B, L, D),
            y_tb.reshape(B, L, D),
            y_bt.reshape(B, L, D),
        ], dim=-1)

        y = self.merge(y)
        y = self.norm(y)

        return y


class MambaBlock(nn.Module):
    """
    Mamba block for vision with cross-scan for 2D context.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_cross_scan: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.use_cross_scan = use_cross_scan

        self.norm = nn.LayerNorm(dim)

        if use_cross_scan:
            self.ssm = CrossScanMamba(dim, d_state, d_conv, expand)
        else:
            self.ssm = BidirectionalMamba(dim, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor, H: Optional[int] = None, W: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input features
            H, W: spatial dimensions (required for cross-scan)

        Returns:
            [B, L, D] output features
        """
        residual = x
        x = self.norm(x)

        if self.use_cross_scan and H is not None and W is not None:
            x = self.ssm(x, H, W)
        else:
            x = self.ssm(x)

        return x + residual


class MambaStage(nn.Module):
    """
    Mamba stage with multiple blocks and downsampling.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int = 2,
        downsample: bool = True,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Downsampling
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
                nn.GroupNorm(1, out_dim),  # Use GroupNorm instead of LayerNorm for spatial
            )
        else:
            if in_dim != out_dim:
                self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1)
            else:
                self.downsample = nn.Identity()

        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=out_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_cross_scan=True,
            )
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input features

        Returns:
            [B, C', H', W'] output features
        """
        # Downsample
        x = self.downsample(x)
        B, C, H, W = x.shape

        # Flatten to sequence
        x = rearrange(x, "b c h w -> b (h w) c")

        # Apply Mamba blocks with spatial info
        for block in self.blocks:
            x = block(x, H, W)

        # Reshape back to spatial
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x


class MambaBackbone(nn.Module):
    """
    Hierarchical Mamba backbone for object detection.

    Uses state space models for O(n) global context with cross-scanning
    for 2D spatial awareness.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dims: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        if dims is None:
            dims = [96, 192, 384, 768]
        if depths is None:
            depths = [2, 2, 6, 2]
        super().__init__()

        # Stem: patch embedding
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0]),
        )

        # Stages
        self.stages = nn.ModuleList()
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            in_dim = dims[i - 1] if i > 0 else dims[0]
            stage = MambaStage(
                in_dim=in_dim,
                out_dim=dim,
                depth=depth,
                downsample=(i > 0),
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.stages.append(stage)

        self.dims = dims
        self.num_stages = len(dims)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] input image

        Returns:
            List of multi-scale features:
            - [B, dims[0], H/4, W/4]
            - [B, dims[1], H/8, W/8]
            - [B, dims[2], H/16, W/16]
            - [B, dims[3], H/32, W/32]
        """
        x = self.stem(x)

        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features
