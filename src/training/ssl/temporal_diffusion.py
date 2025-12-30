"""
Temporal-Diffusion Consistency (TDC) - Novel Video Self-Supervision.

Core Insight:
For video, we can combine temporal consistency with diffusion dynamics:
1. Diffusion convergence should be consistent across frames
2. Warped boxes from frame t should guide diffusion in frame t+1
3. The "attractor landscape" of diffusion should be temporally smooth

This is fundamentally different from:
- Standard temporal consistency (which just matches detections)
- Optical flow supervision (which uses motion as labels)

We're using DIFFUSION TRAJECTORY CONSISTENCY as the supervision signal.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Type alias for COFNet model (avoid circular imports)
COFNetModel = Any


class TemporalDiffusionConsistency(nn.Module):
    """
    Enforce consistency of diffusion dynamics across video frames.

    Key ideas:
    1. Trajectory Consistency: Diffusion paths should warp consistently
    2. Attractor Consistency: Objects (attractors) should move smoothly
    3. Flow-Guided Initialization: Use optical flow to initialize diffusion
    """

    def __init__(
        self,
        trajectory_weight: float = 1.0,
        attractor_weight: float = 1.0,
        flow_init_weight: float = 0.5,
        num_diffusion_steps: int = 8,
    ):
        super().__init__()
        self.trajectory_weight = trajectory_weight
        self.attractor_weight = attractor_weight
        self.flow_init_weight = flow_init_weight
        self.num_diffusion_steps = num_diffusion_steps

        # Lightweight flow estimator (or use external like RAFT)
        self.flow_estimator = LightweightFlowNet()

    def forward(
        self,
        model: COFNetModel,
        frame_t: torch.Tensor,
        frame_t1: torch.Tensor,
        flow: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute temporal-diffusion consistency losses.

        Args:
            model: COFNet model
            frame_t: [B, 3, H, W] frame at time t
            frame_t1: [B, 3, H, W] frame at time t+1
            flow: Optional [B, 2, H, W] optical flow from t to t+1

        Returns:
            Dict containing losses
        """
        B, _, H, W = frame_t.shape
        device = frame_t.device

        # Estimate flow if not provided
        if flow is None:
            flow = self.flow_estimator(frame_t, frame_t1)

        # At this point flow is guaranteed to be a Tensor
        assert flow is not None  # For type checker

        # Extract features for both frames
        features_t = model.backbone(frame_t)
        features_t1 = model.backbone(frame_t1)

        csf_t = model.csf(features_t)
        csf_t1 = model.csf(features_t1)

        losses = {}

        # 1. Diffusion Trajectory Consistency
        losses['trajectory'] = self._trajectory_consistency_loss(
            model, csf_t, csf_t1, flow
        ) * self.trajectory_weight

        # 2. Attractor Consistency Loss
        losses['attractor'] = self._attractor_consistency_loss(
            model, csf_t, csf_t1, flow
        ) * self.attractor_weight

        # 3. Flow-Guided Initialization Loss
        losses['flow_init'] = self._flow_guided_init_loss(
            model, csf_t, csf_t1, flow
        ) * self.flow_init_weight

        losses['total'] = sum(losses.values())

        return losses

    def _trajectory_consistency_loss(
        self,
        model: COFNetModel,
        csf_t: torch.Tensor,
        csf_t1: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """
        Diffusion trajectories should be consistent when warped.

        If we run diffusion on frame t and warp the intermediate boxes
        to frame t+1, they should match diffusion run on frame t+1.
        """
        B = csf_t.shape[0]
        device = csf_t.device
        num_queries = 50

        # Initialize same random boxes for both frames
        init_boxes = torch.rand(B, num_queries, 4, device=device)

        # Run diffusion on frame t and record trajectory
        trajectory_t = self._run_diffusion_with_trajectory(
            model, init_boxes, csf_t
        )

        # Warp trajectory to frame t+1
        warped_trajectory = self._warp_trajectory(trajectory_t, flow)

        # Run diffusion on frame t+1 from warped initialization
        warped_init = warped_trajectory[-1]  # Use final position as init
        trajectory_t1 = self._run_diffusion_with_trajectory(
            model, warped_init, csf_t1
        )

        # Loss: trajectories should match
        # Compare intermediate steps, not just final
        total_loss = torch.tensor(0.0, device=device)
        num_steps = min(len(trajectory_t), len(trajectory_t1))

        for step in range(num_steps):
            boxes_t_warped = self._warp_boxes(trajectory_t[step], flow)
            boxes_t1 = trajectory_t1[step]

            # L1 loss between warped and actual
            loss = F.l1_loss(boxes_t_warped, boxes_t1)
            total_loss = total_loss + loss

        return total_loss / num_steps

    def _attractor_consistency_loss(
        self,
        model: COFNetModel,
        csf_t: torch.Tensor,
        csf_t1: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """
        Object "attractors" should move smoothly according to flow.

        We find where diffusion converges (attractors) and check that
        they move consistently with optical flow.
        """
        B = csf_t.shape[0]
        device = csf_t.device
        num_queries = 100

        # Find attractors in frame t
        init_boxes_t = torch.rand(B, num_queries, 4, device=device)
        converged_t = model.box_refiner.sample(
            init_boxes_t, csf_t, num_steps=self.num_diffusion_steps
        )

        # Find attractors in frame t+1
        init_boxes_t1 = torch.rand(B, num_queries, 4, device=device)
        converged_t1 = model.box_refiner.sample(
            init_boxes_t1, csf_t1, num_steps=self.num_diffusion_steps
        )

        # Warp attractors from t to t+1
        warped_attractors = self._warp_boxes(converged_t, flow)

        # Match warped attractors to actual attractors
        loss = self._hungarian_matching_loss(warped_attractors, converged_t1)

        return loss

    def _flow_guided_init_loss(
        self,
        model: COFNetModel,
        csf_t: torch.Tensor,
        csf_t1: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flow-guided initialization should lead to faster convergence.

        If we initialize diffusion at frame t+1 using warped boxes
        from frame t, it should converge faster (fewer steps) than
        random initialization.
        """
        B = csf_t.shape[0]
        device = csf_t.device
        num_queries = 50

        # Get converged boxes from frame t
        init_t = torch.rand(B, num_queries, 4, device=device)
        converged_t = model.box_refiner.sample(
            init_t, csf_t, num_steps=self.num_diffusion_steps
        )

        # Warp to frame t+1 as initialization
        warped_init = self._warp_boxes(converged_t, flow)

        # Run minimal diffusion (should already be close)
        refined = model.box_refiner.sample(
            warped_init, csf_t1, num_steps=2  # Very few steps
        )

        # Compare to full diffusion from random init
        random_init = torch.rand(B, num_queries, 4, device=device)
        random_refined = model.box_refiner.sample(
            random_init, csf_t1, num_steps=self.num_diffusion_steps
        )

        # Loss: flow-guided should match full refinement
        # This encourages learning to use temporal information
        loss = self._hungarian_matching_loss(refined, random_refined)

        return loss

    def _run_diffusion_with_trajectory(
        self,
        model: COFNetModel,
        init_boxes: torch.Tensor,
        features: torch.Tensor,
    ) -> list:
        """
        Run diffusion and return intermediate boxes at each step.
        """
        trajectory = [init_boxes.clone()]

        B, N, _ = init_boxes.shape
        device = init_boxes.device

        # Flatten features if spatial
        if features.dim() == 4:
            features = features.flatten(2).transpose(1, 2)

        x_t = init_boxes.clone()
        num_steps = self.num_diffusion_steps
        step_size = model.box_refiner.num_steps // num_steps
        timesteps = list(range(0, model.box_refiner.num_steps, step_size))[::-1]

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict noise
            with torch.no_grad():
                noise_pred = model.box_refiner.denoiser(x_t, features, t_batch)

            # DDIM update
            alphas_cumprod = model.box_refiner.alphas_cumprod
            alpha_t = alphas_cumprod[t]

            if i < len(timesteps) - 1:
                alpha_prev = alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=device)

            x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x_0_pred = x_0_pred.clamp(0, 1)

            if i < len(timesteps) - 1:
                x_t = torch.sqrt(alpha_prev) * x_0_pred + torch.sqrt(1 - alpha_prev) * noise_pred
            else:
                x_t = x_0_pred

            trajectory.append(x_t.clone())

        return trajectory

    def _warp_trajectory(
        self,
        trajectory: list,
        flow: torch.Tensor,
    ) -> list:
        """Warp all boxes in a trajectory using optical flow."""
        return [self._warp_boxes(boxes, flow) for boxes in trajectory]

    def _warp_boxes(
        self,
        boxes: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """
        Warp boxes using optical flow.

        Args:
            boxes: [B, N, 4] boxes in (cx, cy, w, h) normalized format
            flow: [B, 2, H, W] optical flow

        Returns:
            [B, N, 4] warped boxes
        """
        B, N, _ = boxes.shape
        _, _, H, W = flow.shape
        device = boxes.device

        # Get box centers in pixel coordinates
        cx = boxes[:, :, 0] * W  # [B, N]
        cy = boxes[:, :, 1] * H

        # Sample flow at box centers
        # Create grid for sampling
        grid_x = (cx / W) * 2 - 1  # Normalize to [-1, 1]
        grid_y = (cy / H) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, N, 2]
        grid = grid.unsqueeze(2)  # [B, N, 1, 2]

        # Sample flow
        flow_at_boxes = F.grid_sample(
            flow, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )  # [B, 2, N, 1]

        flow_at_boxes = flow_at_boxes.squeeze(-1).permute(0, 2, 1)  # [B, N, 2]

        # Apply flow to box centers
        new_cx = (cx + flow_at_boxes[:, :, 0]) / W
        new_cy = (cy + flow_at_boxes[:, :, 1]) / H

        # Clamp to valid range
        new_cx = new_cx.clamp(0, 1)
        new_cy = new_cy.clamp(0, 1)

        # Keep width and height (could also scale based on flow divergence)
        warped_boxes = boxes.clone()
        warped_boxes[:, :, 0] = new_cx
        warped_boxes[:, :, 1] = new_cy

        return warped_boxes

    def _hungarian_matching_loss(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute matching loss between two sets of boxes.

        Uses soft matching for differentiability.
        """
        B, N1, _ = boxes1.shape
        _, N2, _ = boxes2.shape
        device = boxes1.device

        total_loss = torch.tensor(0.0, device=device)

        for b in range(B):
            # Compute cost matrix (L1 distance)
            cost = torch.cdist(boxes1[b], boxes2[b], p=1)  # [N1, N2]

            # Soft assignment using Sinkhorn
            assignment = self._sinkhorn(cost)

            # Weighted loss
            loss = (assignment * cost).sum()
            total_loss = total_loss + loss

        return total_loss / B

    def _sinkhorn(
        self,
        cost: torch.Tensor,
        num_iters: int = 10,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Sinkhorn algorithm for soft assignment.
        """
        # Convert to log-domain for stability
        log_K = -cost / temperature

        # Initialize
        log_u = torch.zeros(cost.shape[0], device=cost.device)
        log_v = torch.zeros(cost.shape[1], device=cost.device)

        for _ in range(num_iters):
            log_u = -torch.logsumexp(log_K + log_v[None, :], dim=1)
            log_v = -torch.logsumexp(log_K + log_u[:, None], dim=0)

        # Compute assignment
        assignment = torch.exp(log_K + log_u[:, None] + log_v[None, :])

        return assignment


class LightweightFlowNet(nn.Module):
    """
    Lightweight optical flow estimator for SSL.

    For production, use RAFT or other pretrained flow networks.
    This is a simple fallback for self-contained training.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(6, hidden_dim, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Flow decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, 2, 4, stride=2, padding=1),
        )

    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate optical flow from frame1 to frame2.

        Args:
            frame1, frame2: [B, 3, H, W] frames

        Returns:
            [B, 2, H, W] optical flow
        """
        # Concatenate frames
        x = torch.cat([frame1, frame2], dim=1)  # [B, 6, H, W]

        # Encode
        features = self.encoder(x)

        # Decode flow
        flow = self.decoder(features)

        # Resize to input resolution if needed
        if flow.shape[2:] != frame1.shape[2:]:
            flow = F.interpolate(
                flow, size=frame1.shape[2:],
                mode='bilinear', align_corners=False
            )

        return flow
