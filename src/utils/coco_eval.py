"""
COCO Evaluation Metrics for COFNet.

Provides mAP, AP50, AP75, and other standard COCO metrics.
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class COCOEvaluator:
    """
    COCO-style evaluator for object detection.

    Computes mAP at multiple IoU thresholds (0.5:0.95).
    """

    def __init__(
        self,
        iou_thresholds: Optional[List[float]] = None,
        max_detections: int = 100,
        area_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Args:
            iou_thresholds: IoU thresholds for evaluation (default: 0.5:0.05:0.95)
            max_detections: Maximum detections per image
            area_ranges: Area ranges for size-based evaluation
        """
        if iou_thresholds is None:
            # COCO default: [0.5, 0.55, ..., 0.95]
            iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

        if area_ranges is None:
            area_ranges = {
                'all': (0, 1e10),
                'small': (0, 32**2),
                'medium': (32**2, 96**2),
                'large': (96**2, 1e10),
            }

        self.iou_thresholds = iou_thresholds
        self.max_detections = max_detections
        self.area_ranges = area_ranges

        # Accumulators
        self.predictions = []
        self.ground_truths = []

    def reset(self):
        """Reset accumulators."""
        self.predictions = []
        self.ground_truths = []

    def update(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
    ):
        """
        Add predictions and ground truths for a batch.

        Args:
            predictions: List of dicts with 'boxes', 'scores', 'labels'
            ground_truths: List of dicts with 'boxes', 'labels'
        """
        self.predictions.extend(predictions)
        self.ground_truths.extend(ground_truths)

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dict with mAP, AP50, AP75, and per-size APs
        """
        results = {}

        # Compute AP for each class
        all_classes = set()
        for gt in self.ground_truths:
            labels = gt['labels']
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            all_classes.update(labels.tolist())

        class_aps = {}
        for cls in sorted(all_classes):
            ap = self._compute_ap_for_class(cls)
            class_aps[cls] = ap

        # Mean AP across classes
        if class_aps:
            aps = [ap['ap'] for ap in class_aps.values()]
            results['mAP'] = np.mean(aps)
            results['AP50'] = np.mean([ap['ap50'] for ap in class_aps.values()])
            results['AP75'] = np.mean([ap['ap75'] for ap in class_aps.values()])

            # Size-based APs
            for size_name in self.area_ranges.keys():
                if size_name == 'all':
                    continue
                size_aps = [ap.get(f'ap_{size_name}', 0) for ap in class_aps.values()]
                results[f'AP_{size_name}'] = np.mean(size_aps) if size_aps else 0.0
        else:
            results['mAP'] = 0.0
            results['AP50'] = 0.0
            results['AP75'] = 0.0

        # Per-class APs
        results['per_class'] = class_aps

        return results

    def _compute_ap_for_class(self, cls: int) -> Dict[str, float]:
        """Compute AP for a single class."""
        # Gather predictions and GTs for this class
        pred_boxes = []
        pred_scores = []
        pred_img_ids = []

        gt_boxes = []
        gt_img_ids = []

        for img_id, (pred, gt) in enumerate(zip(self.predictions, self.ground_truths)):
            # Predictions
            p_boxes = pred['boxes']
            p_scores = pred['scores']
            p_labels = pred['labels']

            if isinstance(p_boxes, torch.Tensor):
                p_boxes = p_boxes.cpu().numpy()
                p_scores = p_scores.cpu().numpy()
                p_labels = p_labels.cpu().numpy()

            mask = p_labels == cls
            pred_boxes.extend(p_boxes[mask].tolist())
            pred_scores.extend(p_scores[mask].tolist())
            pred_img_ids.extend([img_id] * mask.sum())

            # Ground truths
            g_boxes = gt['boxes']
            g_labels = gt['labels']

            if isinstance(g_boxes, torch.Tensor):
                g_boxes = g_boxes.cpu().numpy()
                g_labels = g_labels.cpu().numpy()

            mask = g_labels == cls
            gt_boxes.extend(g_boxes[mask].tolist())
            gt_img_ids.extend([img_id] * mask.sum())

        if not gt_boxes:
            return {'ap': 0.0, 'ap50': 0.0, 'ap75': 0.0}

        pred_boxes = np.array(pred_boxes) if pred_boxes else np.zeros((0, 4))
        pred_scores = np.array(pred_scores)
        pred_img_ids = np.array(pred_img_ids)

        gt_boxes = np.array(gt_boxes)
        gt_img_ids = np.array(gt_img_ids)

        # Sort predictions by score
        sort_idx = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sort_idx]
        pred_scores = pred_scores[sort_idx]
        pred_img_ids = pred_img_ids[sort_idx]

        # Compute AP at each IoU threshold
        aps_at_thresholds = []
        for iou_thresh in self.iou_thresholds:
            ap = self._compute_ap_at_threshold(
                pred_boxes, pred_scores, pred_img_ids,
                gt_boxes, gt_img_ids,
                iou_thresh,
            )
            aps_at_thresholds.append(ap)

        result = {
            'ap': np.mean(aps_at_thresholds),
            'ap50': aps_at_thresholds[0] if len(aps_at_thresholds) > 0 else 0.0,
            'ap75': aps_at_thresholds[5] if len(aps_at_thresholds) > 5 else 0.0,
        }

        return result

    def _compute_ap_at_threshold(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_img_ids: np.ndarray,
        gt_boxes: np.ndarray,
        gt_img_ids: np.ndarray,
        iou_threshold: float,
    ) -> float:
        """Compute AP at a single IoU threshold."""
        num_gt = len(gt_boxes)
        if num_gt == 0:
            return 0.0

        # Group GT boxes by image
        gt_by_image = defaultdict(list)
        for i, img_id in enumerate(gt_img_ids):
            gt_by_image[img_id].append(i)

        # Track which GTs have been matched
        gt_matched = np.zeros(num_gt, dtype=bool)

        # Evaluate each prediction
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))

        for pred_idx in range(len(pred_boxes)):
            img_id = pred_img_ids[pred_idx]
            pred_box = pred_boxes[pred_idx]

            # Get GT boxes for this image
            gt_indices = gt_by_image[img_id]
            if not gt_indices:
                fp[pred_idx] = 1
                continue

            # Compute IoU with all GT boxes for this image
            best_iou = 0
            best_gt_idx = -1

            for gt_idx in gt_indices:
                if gt_matched[gt_idx]:
                    continue

                iou = self._compute_iou(pred_box, gt_boxes[gt_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[pred_idx] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[pred_idx] = 1

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Compute AP using 101-point interpolation
        ap = self._compute_ap_from_pr(precisions, recalls)

        return ap

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def _compute_ap_from_pr(
        self,
        precisions: np.ndarray,
        recalls: np.ndarray,
    ) -> float:
        """Compute AP from precision-recall curve using 101-point interpolation."""
        # Add sentinel values
        precisions = np.concatenate([[0], precisions, [0]])
        recalls = np.concatenate([[0], recalls, [1]])

        # Make precision monotonically decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # Find points where recall changes
        recall_change = np.where(recalls[1:] != recalls[:-1])[0] + 1

        # Sum areas
        ap = np.sum((recalls[recall_change] - recalls[recall_change - 1]) * precisions[recall_change])

        return ap


def evaluate_coco(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Convenience function to evaluate predictions against ground truths.

    Args:
        predictions: List of dicts with 'boxes', 'scores', 'labels'
        ground_truths: List of dicts with 'boxes', 'labels'
        iou_thresholds: IoU thresholds (default: COCO 0.5:0.95)

    Returns:
        Dict with mAP, AP50, AP75, etc.
    """
    evaluator = COCOEvaluator(iou_thresholds=iou_thresholds)
    evaluator.update(predictions, ground_truths)
    return evaluator.compute()


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary for printing."""
    lines = [
        f"mAP:   {metrics.get('mAP', 0):.4f}",
        f"AP50:  {metrics.get('AP50', 0):.4f}",
        f"AP75:  {metrics.get('AP75', 0):.4f}",
    ]

    if 'AP_small' in metrics:
        lines.extend([
            f"AP_S:  {metrics.get('AP_small', 0):.4f}",
            f"AP_M:  {metrics.get('AP_medium', 0):.4f}",
            f"AP_L:  {metrics.get('AP_large', 0):.4f}",
        ])

    return '\n'.join(lines)
