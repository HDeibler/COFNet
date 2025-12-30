"""
Diffusion Convergence Discovery (DCD) - Novel Self-Supervised Object Discovery.

Core Insight:
When we run the diffusion denoising process multiple times from different random
initializations, boxes that consistently converge to the same spatial locations
are likely to be objects. This uses the diffusion dynamics as an implicit
saliency detector WITHOUT any labels.

How it works:
1. Initialize K sets of random boxes
2. Run diffusion denoising on each set (with same image features)
3. Cluster the converged box locations
4. High-density clusters = likely objects (pseudo-labels)
5. Train the model to detect these discovered objects

This is fundamentally different from:
- Contrastive learning (which learns features, not detections)
- Motion-based SSL (which needs video)
- Masked image modeling (which doesn't discover objects)

We're using the STRUCTURE of the detection problem (iterative refinement)
as self-supervision.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# Type alias for COFNet model (avoid circular imports)
COFNetModel = Any


class DiffusionConvergenceDiscovery(nn.Module):
    """
    Discover objects by analyzing where diffusion boxes converge.

    The key insight: objects create "attractors" in the diffusion landscape.
    Random boxes will consistently converge toward object locations.
    """

    def __init__(
        self,
        num_initializations: int = 8,
        convergence_threshold: float = 0.1,
        min_cluster_size: int = 3,
        objectness_temperature: float = 0.5,
    ):
        super().__init__()
        self.num_initializations = num_initializations
        self.convergence_threshold = convergence_threshold
        self.min_cluster_size = min_cluster_size
        self.objectness_temperature = objectness_temperature

    def forward(
        self,
        model: COFNetModel,
        images: torch.Tensor,
        num_queries: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """
        Discover objects via diffusion convergence analysis.

        Args:
            model: COFNet model (or just the backbone + CSF + diffusion parts)
            images: [B, 3, H, W] input images
            num_queries: Number of box queries per initialization

        Returns:
            Dict containing:
                - discovered_boxes: [B, N, 4] pseudo-label boxes
                - objectness_scores: [B, N] confidence scores
                - convergence_maps: [B, H', W'] spatial objectness heatmap
        """
        B = images.shape[0]
        device = images.device

        # Extract features once (shared across all initializations)
        with torch.no_grad():
            backbone_features = model.backbone(images)
            csf_features = model.csf(backbone_features)

        # Run multiple diffusion trajectories
        all_converged_boxes = []

        for k in range(self.num_initializations):
            # Random initialization
            init_boxes = torch.rand(B, num_queries, 4, device=device)

            # Run diffusion to convergence
            with torch.no_grad():
                converged = model.box_refiner.sample(
                    init_boxes,
                    csf_features,
                    num_steps=8,
                )
            all_converged_boxes.append(converged)

        # Stack all converged boxes: [B, K, N, 4]
        all_boxes = torch.stack(all_converged_boxes, dim=1)

        # Analyze convergence patterns
        discovered_boxes, objectness_scores = self._cluster_convergences(all_boxes)

        # Generate spatial objectness map
        img_h, img_w = images.shape[2], images.shape[3]
        convergence_maps = self._compute_convergence_map(
            all_boxes, (img_h, img_w)
        )

        return {
            'discovered_boxes': discovered_boxes,
            'objectness_scores': objectness_scores,
            'convergence_maps': convergence_maps,
        }

    def _cluster_convergences(
        self,
        all_boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cluster converged boxes to find consistent object locations.

        Boxes from different initializations that converge to similar
        locations indicate high objectness.

        Args:
            all_boxes: [B, K, N, 4] boxes from K initializations

        Returns:
            discovered_boxes: [B, M, 4] clustered box centers
            objectness_scores: [B, M] confidence based on cluster density
        """
        B, K, N, _ = all_boxes.shape
        device = all_boxes.device

        discovered_boxes_list = []
        objectness_list = []

        for b in range(B):
            # Flatten all boxes for this image: [K*N, 4]
            boxes_flat = all_boxes[b].reshape(-1, 4)

            # Compute pairwise IoU
            ious = self._box_iou(boxes_flat, boxes_flat)

            # Find clusters using greedy clustering
            clusters = self._greedy_cluster(ious, self.convergence_threshold)

            # Extract cluster centers and compute objectness
            centers = []
            scores = []

            for cluster_indices in clusters:
                if len(cluster_indices) >= self.min_cluster_size:
                    # Cluster center = mean of boxes in cluster
                    cluster_boxes = boxes_flat[cluster_indices]
                    center = cluster_boxes.mean(dim=0)
                    centers.append(center)

                    # Objectness = normalized cluster size
                    # More convergence = higher confidence
                    score = len(cluster_indices) / (K * N)
                    score = torch.tensor(score, device=device)
                    scores.append(score)

            if len(centers) > 0:
                discovered_boxes_list.append(torch.stack(centers))
                objectness_list.append(torch.stack(scores))
            else:
                # No objects discovered - return empty
                discovered_boxes_list.append(torch.zeros(0, 4, device=device))
                objectness_list.append(torch.zeros(0, device=device))

        # Pad to same length
        max_objects = max(len(boxes) for boxes in discovered_boxes_list)
        if max_objects == 0:
            max_objects = 1

        padded_boxes = torch.zeros(B, max_objects, 4, device=device)
        padded_scores = torch.zeros(B, max_objects, device=device)

        for b, (boxes, scores) in enumerate(zip(discovered_boxes_list, objectness_list)):
            if len(boxes) > 0:
                n = min(len(boxes), max_objects)
                padded_boxes[b, :n] = boxes[:n]
                padded_scores[b, :n] = scores[:n]

        return padded_boxes, padded_scores

    def _greedy_cluster(
        self,
        similarity: torch.Tensor,
        threshold: float,
    ) -> List[List[int]]:
        """
        Greedy clustering based on similarity matrix.
        """
        N = similarity.shape[0]
        assigned = [False] * N
        clusters = []

        # Sort by number of neighbors (most connected first)
        neighbor_counts = (similarity > threshold).sum(dim=1)
        order = neighbor_counts.argsort(descending=True)

        for i in order.tolist():
            if assigned[i]:
                continue

            # Start new cluster
            cluster = [i]
            assigned[i] = True

            # Add all unassigned neighbors
            neighbors = (similarity[i] > threshold).nonzero(as_tuple=True)[0]
            for j in neighbors.tolist():
                if not assigned[j]:
                    cluster.append(j)
                    assigned[j] = True

            clusters.append(cluster)

        return clusters

    def _box_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute IoU between two sets of boxes in (cx, cy, w, h) format.
        """
        # Convert to (x1, y1, x2, y2)
        b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
        b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
        b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
        b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2

        b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
        b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
        b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
        b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2

        # Intersection
        inter_x1 = torch.max(b1_x1[:, None], b2_x1[None, :])
        inter_y1 = torch.max(b1_y1[:, None], b2_y1[None, :])
        inter_x2 = torch.min(b1_x2[:, None], b2_x2[None, :])
        inter_y2 = torch.min(b1_y2[:, None], b2_y2[None, :])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        union_area = area1[:, None] + area2[None, :] - inter_area

        return inter_area / union_area.clamp(min=1e-6)

    def _compute_convergence_map(
        self,
        all_boxes: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Create spatial heatmap of where boxes tend to converge.

        This is useful for visualization and can serve as a
        soft objectness prior for the detection head.
        """
        B, K, N, _ = all_boxes.shape
        device = all_boxes.device
        H, W = image_size

        # Create lower-res heatmap
        map_h, map_w = H // 16, W // 16
        heatmaps = torch.zeros(B, map_h, map_w, device=device)

        for b in range(B):
            for k in range(K):
                for n in range(N):
                    cx, cy, w, h = all_boxes[b, k, n]

                    # Convert to heatmap coordinates
                    hx = int(cx * map_w)
                    hy = int(cy * map_h)

                    # Clamp to valid range
                    hx = max(0, min(hx, map_w - 1))
                    hy = max(0, min(hy, map_h - 1))

                    # Add gaussian blob
                    heatmaps[b] = self._add_gaussian(
                        heatmaps[b], hx, hy, sigma=2.0
                    )

        # Normalize
        heatmaps = heatmaps / (K * N)

        return heatmaps

    def _add_gaussian(
        self,
        heatmap: torch.Tensor,
        cx: int,
        cy: int,
        sigma: float = 2.0,
    ) -> torch.Tensor:
        """Add a Gaussian blob to heatmap at (cx, cy)."""
        H, W = heatmap.shape
        device = heatmap.device

        # Create coordinate grids
        y = torch.arange(H, device=device).float()
        x = torch.arange(W, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        # Gaussian
        gaussian = torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))

        return heatmap + gaussian


class DiffusionConvergenceLoss(nn.Module):
    """
    Loss function for training with discovered pseudo-labels.

    After discovering objects via convergence analysis, we train
    the model to detect them consistently.
    """

    def __init__(
        self,
        consistency_weight: float = 1.0,
        diversity_weight: float = 0.1,
    ):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.diversity_weight = diversity_weight

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        discoveries: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute loss between model predictions and discovered pseudo-labels.

        Args:
            predictions: Model outputs (pred_boxes, pred_logits)
            discoveries: From DiffusionConvergenceDiscovery

        Returns:
            Combined loss
        """
        pred_boxes = predictions['pred_boxes']
        discovered_boxes = discoveries['discovered_boxes']
        objectness_scores = discoveries['objectness_scores']

        B = pred_boxes.shape[0]
        device = pred_boxes.device
        total_loss = torch.tensor(0.0, device=device)

        for b in range(B):
            # Only use discovered boxes with sufficient confidence
            valid_mask = objectness_scores[b] > 0.1
            if not valid_mask.any():
                continue

            gt_boxes = discovered_boxes[b][valid_mask]
            weights = objectness_scores[b][valid_mask]

            # Match predictions to discovered boxes
            loss_b = self._matching_loss(
                pred_boxes[b],
                gt_boxes,
                weights,
            )
            total_loss = total_loss + loss_b

        # Diversity loss: encourage predictions to spread out
        diversity_loss = self._diversity_loss(pred_boxes)

        return (
            self.consistency_weight * total_loss / B +
            self.diversity_weight * diversity_loss
        )

    def _matching_loss(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """L1 loss with Hungarian matching."""
        N_pred = pred_boxes.shape[0]
        N_gt = gt_boxes.shape[0]

        if N_gt == 0:
            return torch.tensor(0.0, device=pred_boxes.device)

        # Compute cost matrix
        cost = torch.cdist(pred_boxes, gt_boxes, p=1)  # [N_pred, N_gt]

        # Hungarian matching
        cost_np = cost.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        # Compute weighted loss
        loss = torch.tensor(0.0, device=pred_boxes.device)
        for pred_idx, gt_idx in zip(row_ind, col_ind):
            box_loss = F.l1_loss(pred_boxes[pred_idx], gt_boxes[gt_idx])
            loss = loss + weights[gt_idx] * box_loss

        return loss / max(len(row_ind), 1)

    def _diversity_loss(
        self,
        pred_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage diverse predictions (avoid collapse to same location).

        Uses negative pairwise distance as loss.
        """
        B, N, _ = pred_boxes.shape

        # Compute pairwise distances
        boxes_flat = pred_boxes.reshape(B * N, 4)

        # Centers only for efficiency
        centers = pred_boxes[..., :2]  # [B, N, 2]

        total_loss = torch.tensor(0.0, device=pred_boxes.device)

        for b in range(B):
            # Pairwise squared distances
            dists = torch.cdist(centers[b], centers[b], p=2)

            # Encourage minimum distance (avoid collapse)
            # Use soft min to make it differentiable
            min_dist = dists + torch.eye(N, device=dists.device) * 1e6
            min_dist = -torch.logsumexp(-min_dist / 0.1, dim=1).mean()

            total_loss = total_loss + min_dist

        return -total_loss / B  # Negative because we want to maximize distance
