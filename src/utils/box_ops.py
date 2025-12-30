"""
Box Operations for COFNet.

Utility functions for bounding box manipulation, IoU computation, etc.
"""

from typing import Tuple

import torch


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.

    Args:
        boxes: [..., 4] boxes in (cx, cy, w, h) format

    Returns:
        [..., 4] boxes in (x1, y1, x2, y2) format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format.

    Args:
        boxes: [..., 4] boxes in (x1, y1, x2, y2) format

    Returns:
        [..., 4] boxes in (cx, cy, w, h) format
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute area of boxes.

    Args:
        boxes: [..., 4] boxes in (x1, y1, x2, y2) format

    Returns:
        [...] area of each box
    """
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] boxes in (x1, y1, x2, y2) format
        boxes2: [M, 4] boxes in (x1, y1, x2, y2) format

    Returns:
        [N, M] IoU matrix
    """
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Union
    union = area1[:, None] + area2[None, :] - inter

    return inter / union.clamp(min=1e-6)


def box_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] boxes in (x1, y1, x2, y2) format
        boxes2: [M, 4] boxes in (x1, y1, x2, y2) format

    Returns:
        [N, M] GIoU matrix
    """
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Union
    union = area1[:, None] + area2[None, :] - inter

    # IoU
    iou = inter / union.clamp(min=1e-6)

    # Enclosing box
    enc_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    enc_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    enc_wh = enc_rb - enc_lt  # [N, M, 2]
    enc_area = enc_wh[:, :, 0] * enc_wh[:, :, 1]  # [N, M]

    # GIoU
    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)

    return giou


def box_diou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Distance IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] boxes in (x1, y1, x2, y2) format
        boxes2: [M, 4] boxes in (x1, y1, x2, y2) format

    Returns:
        [N, M] DIoU matrix
    """
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Union
    union = area1[:, None] + area2[None, :] - inter

    # IoU
    iou = inter / union.clamp(min=1e-6)

    # Centers
    center1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2  # [N, 2]
    center2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2  # [M, 2]

    # Center distance
    center_dist = ((center1[:, None, :] - center2[None, :, :]) ** 2).sum(dim=-1)  # [N, M]

    # Enclosing box diagonal
    enc_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    enc_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    diag = ((enc_rb - enc_lt) ** 2).sum(dim=-1)  # [N, M]

    # DIoU
    diou = iou - center_dist / diag.clamp(min=1e-6)

    return diou


def box_ciou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Complete IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] boxes in (x1, y1, x2, y2) format
        boxes2: [M, 4] boxes in (x1, y1, x2, y2) format

    Returns:
        [N, M] CIoU matrix
    """
    import math

    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [M]

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Union
    union = area1[:, None] + area2[None, :] - inter

    # IoU
    iou = inter / union.clamp(min=1e-6)

    # Centers
    center1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2  # [N, 2]
    center2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2  # [M, 2]

    # Center distance
    center_dist = ((center1[:, None, :] - center2[None, :, :]) ** 2).sum(dim=-1)  # [N, M]

    # Enclosing box diagonal
    enc_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    enc_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    diag = ((enc_rb - enc_lt) ** 2).sum(dim=-1)  # [N, M]

    # Aspect ratio consistency
    w1 = boxes1[:, 2] - boxes1[:, 0]  # [N]
    h1 = boxes1[:, 3] - boxes1[:, 1]  # [N]
    w2 = boxes2[:, 2] - boxes2[:, 0]  # [M]
    h2 = boxes2[:, 3] - boxes2[:, 1]  # [M]

    v = (4 / (math.pi ** 2)) * (
        torch.atan(w2[None, :] / h2[None, :].clamp(min=1e-6))
        - torch.atan(w1[:, None] / h1[:, None].clamp(min=1e-6))
    ) ** 2  # [N, M]

    alpha = v / ((1 - iou) + v).clamp(min=1e-6)

    # CIoU
    ciou = iou - center_dist / diag.clamp(min=1e-6) - alpha * v

    return ciou


def rescale_boxes(
    boxes: torch.Tensor,
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Rescale boxes from original image size to target size.

    Args:
        boxes: [..., 4] boxes in (x1, y1, x2, y2) format
        original_size: (height, width) of original image
        target_size: (height, width) of target image

    Returns:
        [..., 4] rescaled boxes
    """
    oh, ow = original_size
    th, tw = target_size

    scale_x = tw / ow
    scale_y = th / oh

    boxes = boxes.clone()
    boxes[..., 0] *= scale_x
    boxes[..., 2] *= scale_x
    boxes[..., 1] *= scale_y
    boxes[..., 3] *= scale_y

    return boxes


def clip_boxes(
    boxes: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    """
    Clip boxes to image boundaries.

    Args:
        boxes: [..., 4] boxes in (x1, y1, x2, y2) format
        size: (height, width) of image

    Returns:
        [..., 4] clipped boxes
    """
    h, w = size
    boxes = boxes.clone()
    boxes[..., 0] = boxes[..., 0].clamp(0, w)
    boxes[..., 1] = boxes[..., 1].clamp(0, h)
    boxes[..., 2] = boxes[..., 2].clamp(0, w)
    boxes[..., 3] = boxes[..., 3].clamp(0, h)
    return boxes


def remove_small_boxes(
    boxes: torch.Tensor,
    min_size: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove boxes with width or height less than min_size.

    Args:
        boxes: [N, 4] boxes in (x1, y1, x2, y2) format
        min_size: Minimum size threshold

    Returns:
        Tuple of (filtered_boxes, keep_indices)
    """
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    return boxes[keep], keep


def batched_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute IoU for batched boxes (same number of boxes).

    Args:
        boxes1: [N, 4] boxes in (x1, y1, x2, y2) format
        boxes2: [N, 4] boxes in (x1, y1, x2, y2) format

    Returns:
        [N] IoU for each pair
    """
    area1 = box_area(boxes1)  # [N]
    area2 = box_area(boxes2)  # [N]

    # Intersection
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N, 2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N, 2]

    wh = (rb - lt).clamp(min=0)  # [N, 2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    # Union
    union = area1 + area2 - inter

    return inter / union.clamp(min=1e-6)
