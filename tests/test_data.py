"""
Tests for COFNet data loading and utilities.

Run with: python -m pytest tests/test_data.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestBoxOperations:
    """Test box utility functions."""

    def test_box_format_conversion(self):
        """Test converting between box formats."""
        from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

        # cxcywh format
        boxes_cxcywh = torch.tensor([
            [0.5, 0.5, 0.2, 0.4],  # center at (0.5, 0.5), size 0.2x0.4
            [0.3, 0.7, 0.1, 0.2],
        ])

        # Convert to xyxy
        boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)

        expected_xyxy = torch.tensor([
            [0.4, 0.3, 0.6, 0.7],  # x1=0.5-0.1, y1=0.5-0.2, x2=0.5+0.1, y2=0.5+0.2
            [0.25, 0.6, 0.35, 0.8],
        ])

        assert torch.allclose(boxes_xyxy, expected_xyxy, atol=1e-6)

        # Convert back to cxcywh
        boxes_back = box_xyxy_to_cxcywh(boxes_xyxy)
        assert torch.allclose(boxes_back, boxes_cxcywh, atol=1e-6)

    def test_box_iou(self):
        """Test IoU computation."""
        from utils.box_ops import box_iou

        # Two boxes with 50% overlap
        boxes1 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])  # Area = 4
        boxes2 = torch.tensor([[1.0, 0.0, 3.0, 2.0]])  # Area = 4, overlap = 2

        iou = box_iou(boxes1, boxes2)
        expected = 2.0 / 6.0  # intersection / union = 2 / (4 + 4 - 2)
        assert torch.allclose(iou, torch.tensor([[expected]]), atol=1e-6)

        # No overlap
        boxes3 = torch.tensor([[5.0, 5.0, 6.0, 6.0]])
        iou_no_overlap = box_iou(boxes1, boxes3)
        assert iou_no_overlap.item() == 0.0

        # Perfect overlap
        iou_self = box_iou(boxes1, boxes1)
        assert torch.allclose(iou_self, torch.tensor([[1.0]]))

    def test_box_giou(self):
        """Test GIoU computation."""
        from utils.box_ops import box_giou

        boxes1 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        boxes2 = torch.tensor([[1.0, 0.0, 3.0, 2.0]])

        giou = box_giou(boxes1, boxes2)

        # GIoU should be less than or equal to IoU
        from utils.box_ops import box_iou
        iou = box_iou(boxes1, boxes2)

        assert giou.item() <= iou.item()
        assert giou.item() >= -1.0  # GIoU range is [-1, 1]

    def test_batched_box_iou(self):
        """Test batched IoU (pairwise)."""
        from utils.box_ops import batched_box_iou

        boxes1 = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0, 2.0],
        ])
        boxes2 = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],  # Perfect match with boxes1[0]
            [1.0, 1.0, 3.0, 3.0],  # Partial overlap with boxes1[1]
        ])

        ious = batched_box_iou(boxes1, boxes2)

        assert ious[0].item() == 1.0  # Perfect overlap
        assert 0 < ious[1].item() < 1  # Partial overlap


class TestNMS:
    """Test NMS functions."""

    def test_standard_nms(self):
        """Test standard NMS."""
        from utils.nms import nms

        boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.1, 1.1, 1.1],  # High overlap with first box
            [5.0, 5.0, 6.0, 6.0],  # No overlap
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])

        keep = nms(boxes, scores, iou_threshold=0.5)

        # Should keep first and third (second overlaps too much with first)
        assert len(keep) == 2
        assert 0 in keep.tolist()
        assert 2 in keep.tolist()

    def test_batched_nms(self):
        """Test batched NMS (per-class)."""
        from utils.nms import batched_nms

        boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],  # Class 0
            [0.1, 0.1, 1.1, 1.1],  # Class 0, overlaps
            [0.0, 0.0, 1.0, 1.0],  # Class 1
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        labels = torch.tensor([0, 0, 1])

        keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)

        # Class 0: keep first (0.9 > 0.8)
        # Class 1: keep third
        assert len(keep) == 2

    def test_soft_nms(self):
        """Test Soft-NMS."""
        from utils.nms import soft_nms

        boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.1, 1.1, 1.1],
        ])
        scores = torch.tensor([0.9, 0.8])

        keep, new_scores = soft_nms(boxes, scores, iou_threshold=0.5)

        # Both should be kept but second score should be reduced
        assert len(keep) == 2
        assert new_scores[0] == 0.9  # First unchanged
        assert new_scores[1] < 0.8   # Second reduced due to overlap

    def test_postprocess_detections(self):
        """Test full post-processing pipeline."""
        from utils.nms import postprocess_detections

        B, N, C = 2, 10, 3

        pred_boxes = torch.rand(B, N, 4)  # Random boxes
        pred_logits = torch.randn(B, N, C)

        results = postprocess_detections(
            pred_boxes,
            pred_logits,
            conf_threshold=0.1,
            nms_threshold=0.5,
        )

        assert len(results) == B
        for r in results:
            assert 'boxes' in r
            assert 'scores' in r
            assert 'labels' in r
            assert r['boxes'].shape[-1] == 4
            assert len(r['scores']) == len(r['boxes'])
            assert len(r['labels']) == len(r['boxes'])


class TestCOCOEvaluator:
    """Test COCO evaluation."""

    def test_evaluator_basic(self):
        """Test basic evaluation."""
        from utils.coco_eval import COCOEvaluator

        evaluator = COCOEvaluator()

        # Create synthetic predictions and ground truths
        predictions = [
            {
                'boxes': torch.tensor([[10, 10, 50, 50]]),
                'scores': torch.tensor([0.9]),
                'labels': torch.tensor([0]),
            },
        ]
        ground_truths = [
            {
                'boxes': torch.tensor([[10, 10, 50, 50]]),
                'labels': torch.tensor([0]),
            },
        ]

        evaluator.update(predictions, ground_truths)
        metrics = evaluator.compute()

        assert 'mAP' in metrics
        assert 'AP50' in metrics
        assert 'AP75' in metrics
        assert metrics['mAP'] > 0  # Should have some positive mAP

    def test_evaluator_perfect_match(self):
        """Test with perfect predictions."""
        from utils.coco_eval import COCOEvaluator

        evaluator = COCOEvaluator()

        # Perfect match
        predictions = [
            {
                'boxes': torch.tensor([[0, 0, 100, 100]]),
                'scores': torch.tensor([1.0]),
                'labels': torch.tensor([0]),
            },
        ]
        ground_truths = [
            {
                'boxes': torch.tensor([[0, 0, 100, 100]]),
                'labels': torch.tensor([0]),
            },
        ]

        evaluator.update(predictions, ground_truths)
        metrics = evaluator.compute()

        # Perfect prediction should have mAP = 1.0
        assert metrics['AP50'] == 1.0

    def test_evaluator_no_predictions(self):
        """Test with no predictions."""
        from utils.coco_eval import COCOEvaluator

        evaluator = COCOEvaluator()

        predictions = [
            {
                'boxes': torch.zeros((0, 4)),
                'scores': torch.zeros((0,)),
                'labels': torch.zeros((0,), dtype=torch.long),
            },
        ]
        ground_truths = [
            {
                'boxes': torch.tensor([[10, 10, 50, 50]]),
                'labels': torch.tensor([0]),
            },
        ]

        evaluator.update(predictions, ground_truths)
        metrics = evaluator.compute()

        # No predictions = 0 mAP
        assert metrics['mAP'] == 0.0


class TestAugmentations:
    """Test data augmentations."""

    def test_train_transforms(self):
        """Test training transforms."""
        from data.augmentation import get_train_transforms

        transforms = get_train_transforms(image_size=(640, 640))

        # Create dummy image and boxes
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)

        result = transforms(image=image, bboxes=boxes, labels=labels)

        assert 'image' in result
        assert isinstance(result['image'], torch.Tensor)
        assert result['image'].shape == (3, 640, 640)

    def test_val_transforms(self):
        """Test validation transforms."""
        from data.augmentation import get_val_transforms

        transforms = get_val_transforms(image_size=(640, 640))

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)

        result = transforms(image=image, bboxes=boxes, labels=labels)

        assert 'image' in result
        assert result['image'].shape == (3, 640, 640)


def test_integration():
    """Integration test: model forward pass with post-processing."""
    from models.cofnet import COFNet
    from utils.nms import postprocess_detections

    # Small model for testing
    model = COFNet(
        num_classes=3,
        backbone_dims=[32, 64, 128, 256],
        csf_dim=64,
        num_queries=20,
        diffusion_steps_train=10,
        diffusion_steps_infer=2,
    )
    model.eval()

    # Random input
    images = torch.randn(2, 3, 128, 128)

    with torch.no_grad():
        outputs = model(images)

    # Post-process
    results = postprocess_detections(
        outputs['pred_boxes'],
        outputs['pred_logits'],
        conf_threshold=0.1,
        nms_threshold=0.5,
    )

    assert len(results) == 2
    for r in results:
        if len(r['boxes']) > 0:
            assert r['boxes'].shape[-1] == 4
            assert r['labels'].max() < 3  # 3 classes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
