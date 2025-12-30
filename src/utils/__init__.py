"""
COFNet Utilities.
"""

from .box_ops import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    box_area,
    box_iou,
    box_giou,
    box_diou,
    box_ciou,
    rescale_boxes,
    clip_boxes,
    remove_small_boxes,
    batched_box_iou,
)

from .nms import (
    nms,
    soft_nms,
    diou_nms,
    batched_nms,
    matrix_nms,
    postprocess_detections,
)

from .coco_eval import (
    COCOEvaluator,
    evaluate_coco,
    format_metrics,
)

__all__ = [
    # Box operations
    'box_cxcywh_to_xyxy',
    'box_xyxy_to_cxcywh',
    'box_area',
    'box_iou',
    'box_giou',
    'box_diou',
    'box_ciou',
    'rescale_boxes',
    'clip_boxes',
    'remove_small_boxes',
    'batched_box_iou',
    # NMS
    'nms',
    'soft_nms',
    'diou_nms',
    'batched_nms',
    'matrix_nms',
    'postprocess_detections',
    # Evaluation
    'COCOEvaluator',
    'evaluate_coco',
    'format_metrics',
]
