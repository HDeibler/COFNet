"""
Non-Maximum Suppression (NMS) utilities for COFNet.

Includes standard NMS, Soft-NMS, DIoU-NMS, and batched variants.
"""

from typing import List, Optional, Tuple

import torch

from .box_ops import box_iou, box_diou


def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Standard Non-Maximum Suppression.

    Args:
        boxes: [N, 4] boxes in (x1, y1, x2, y2) format
        scores: [N] confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of kept boxes
    """
    # Use torchvision if available (faster)
    try:
        from torchvision.ops import nms as tv_nms
        return tv_nms(boxes, scores, iou_threshold)
    except ImportError:
        pass

    # Fallback to pure PyTorch implementation
    return _nms_pytorch(boxes, scores, iou_threshold)


def _nms_pytorch(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Pure PyTorch NMS implementation."""
    # Sort by scores
    order = scores.argsort(descending=True)

    keep = []
    while len(order) > 0:
        # Pick the box with highest score
        idx = order[0].item()
        keep.append(idx)

        if len(order) == 1:
            break

        # Compute IoU with remaining boxes
        current_box = boxes[idx:idx+1]  # [1, 4]
        remaining_boxes = boxes[order[1:]]  # [M, 4]

        ious = box_iou(current_box, remaining_boxes).squeeze(0)  # [M]

        # Keep boxes with IoU below threshold
        mask = ious <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    method: str = 'gaussian',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Soft Non-Maximum Suppression.

    Instead of removing overlapping boxes, reduce their scores based on IoU.

    Args:
        boxes: [N, 4] boxes in (x1, y1, x2, y2) format
        scores: [N] confidence scores
        iou_threshold: IoU threshold for hard suppression (linear method)
        sigma: Sigma for Gaussian decay
        score_threshold: Minimum score to keep
        method: 'gaussian' or 'linear'

    Returns:
        Tuple of (kept_indices, updated_scores)
    """
    N = boxes.shape[0]
    indices = torch.arange(N, device=boxes.device)
    scores = scores.clone()

    keep = []
    kept_scores = []

    while len(indices) > 0:
        # Find max score
        max_idx = scores[indices].argmax()
        max_pos = indices[max_idx]

        keep.append(max_pos.item())
        kept_scores.append(scores[max_pos].item())

        if len(indices) == 1:
            break

        # Remove current box from indices
        mask = torch.ones(len(indices), dtype=torch.bool, device=boxes.device)
        mask[max_idx] = False
        indices = indices[mask]

        # Compute IoU with remaining boxes
        current_box = boxes[max_pos:max_pos+1]  # [1, 4]
        remaining_boxes = boxes[indices]  # [M, 4]
        ious = box_iou(current_box, remaining_boxes).squeeze(0)  # [M]

        # Update scores
        if method == 'gaussian':
            decay = torch.exp(-(ious ** 2) / sigma)
        else:  # linear
            decay = torch.where(
                ious > iou_threshold,
                1 - ious,
                torch.ones_like(ious),
            )

        scores[indices] *= decay

        # Remove low score boxes
        score_mask = scores[indices] >= score_threshold
        indices = indices[score_mask]

    keep_tensor = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    kept_scores_tensor = torch.tensor(kept_scores, device=boxes.device)

    return keep_tensor, kept_scores_tensor


def diou_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Distance-IoU based NMS.

    Uses DIoU instead of IoU for suppression decisions.

    Args:
        boxes: [N, 4] boxes in (x1, y1, x2, y2) format
        scores: [N] confidence scores
        iou_threshold: DIoU threshold for suppression

    Returns:
        Indices of kept boxes
    """
    order = scores.argsort(descending=True)

    keep = []
    while len(order) > 0:
        idx = order[0].item()
        keep.append(idx)

        if len(order) == 1:
            break

        current_box = boxes[idx:idx+1]  # [1, 4]
        remaining_boxes = boxes[order[1:]]  # [M, 4]

        # Use DIoU instead of IoU
        dious = box_diou(current_box, remaining_boxes).squeeze(0)  # [M]

        mask = dious <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Batched NMS - performs NMS independently for each class.

    Args:
        boxes: [N, 4] boxes in (x1, y1, x2, y2) format
        scores: [N] confidence scores
        labels: [N] class labels
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of kept boxes
    """
    # Use torchvision if available
    try:
        from torchvision.ops import batched_nms as tv_batched_nms
        return tv_batched_nms(boxes, scores, labels, iou_threshold)
    except ImportError:
        pass

    # Fallback: offset boxes by class
    max_coord = boxes.max()
    offsets = labels.float() * (max_coord + 1)
    boxes_offset = boxes + offsets[:, None]

    return nms(boxes_offset, scores, iou_threshold)


def matrix_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    kernel: str = 'gaussian',
    sigma: float = 2.0,
    score_threshold: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Matrix NMS (from SOLO/SOLOv2).

    Efficient parallel NMS using matrix operations.

    Args:
        boxes: [N, 4] boxes in (x1, y1, x2, y2) format
        scores: [N] confidence scores
        labels: [N] class labels
        kernel: 'gaussian' or 'linear'
        sigma: Sigma for Gaussian kernel
        score_threshold: Minimum score to keep

    Returns:
        Tuple of (kept_boxes, kept_scores, kept_labels)
    """
    N = boxes.shape[0]
    if N == 0:
        return boxes, scores, labels

    # Sort by score
    sorted_idx = scores.argsort(descending=True)
    boxes = boxes[sorted_idx]
    scores = scores[sorted_idx]
    labels = labels[sorted_idx]

    # Compute IoU matrix
    ious = box_iou(boxes, boxes)  # [N, N]

    # Zero out lower triangle (only consider higher-scored boxes)
    ious = ious.triu(diagonal=1)

    # Get max IoU for each box (from higher-scored boxes)
    max_iou, _ = ious.max(dim=0)  # [N]

    # Decay factor based on kernel
    if kernel == 'gaussian':
        decay = torch.exp(-(max_iou ** 2) / sigma)
    else:
        decay = 1 - max_iou

    # Apply decay to scores
    scores = scores * decay

    # Filter by score threshold
    keep = scores >= score_threshold

    return boxes[keep], scores[keep], labels[keep]


def postprocess_detections(
    pred_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    conf_threshold: float = 0.05,
    nms_threshold: float = 0.5,
    max_detections: int = 300,
    nms_type: str = 'standard',
) -> List[dict]:
    """
    Post-process COFNet predictions with NMS.

    Args:
        pred_boxes: [B, N, 4] boxes in (cx, cy, w, h) normalized format
        pred_logits: [B, N, C] class logits
        conf_threshold: Minimum confidence threshold
        nms_threshold: IoU threshold for NMS
        max_detections: Maximum detections per image
        nms_type: 'standard', 'soft', 'diou', or 'matrix'

    Returns:
        List of dicts with 'boxes', 'scores', 'labels' per image
    """
    from .box_ops import box_cxcywh_to_xyxy

    B, N, C = pred_logits.shape
    device = pred_boxes.device

    results = []

    for b in range(B):
        boxes = pred_boxes[b]  # [N, 4]
        logits = pred_logits[b]  # [N, C]

        # Get scores and labels
        scores, labels = logits.softmax(dim=-1).max(dim=-1)  # [N], [N]

        # Filter by confidence
        mask = scores >= conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        if len(boxes) == 0:
            results.append({
                'boxes': torch.zeros((0, 4), device=device),
                'scores': torch.zeros((0,), device=device),
                'labels': torch.zeros((0,), dtype=torch.long, device=device),
            })
            continue

        # Convert to xyxy format for NMS
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)

        # Apply NMS
        if nms_type == 'soft':
            keep, scores = soft_nms(boxes_xyxy, scores, nms_threshold)
            boxes_xyxy = boxes_xyxy[keep]
            labels = labels[keep]
        elif nms_type == 'diou':
            keep = diou_nms(boxes_xyxy, scores, nms_threshold)
            boxes_xyxy = boxes_xyxy[keep]
            scores = scores[keep]
            labels = labels[keep]
        elif nms_type == 'matrix':
            boxes_xyxy, scores, labels = matrix_nms(
                boxes_xyxy, scores, labels,
                score_threshold=conf_threshold,
            )
        else:  # standard
            keep = batched_nms(boxes_xyxy, scores, labels, nms_threshold)
            boxes_xyxy = boxes_xyxy[keep]
            scores = scores[keep]
            labels = labels[keep]

        # Limit detections
        if len(boxes_xyxy) > max_detections:
            topk = scores.argsort(descending=True)[:max_detections]
            boxes_xyxy = boxes_xyxy[topk]
            scores = scores[topk]
            labels = labels[topk]

        results.append({
            'boxes': boxes_xyxy,
            'scores': scores,
            'labels': labels,
        })

    return results
