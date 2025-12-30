#!/usr/bin/env python3
"""
COFNet Evaluation Script

Evaluates a trained COFNet model on a COCO-format dataset.

Usage:
    python scripts/eval.py --config configs/cofnet_skywatch.yaml --checkpoint output/best.pth
    python scripts/eval.py --config configs/cofnet_base.yaml --checkpoint output/best.pth --visualize
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.cofnet import COFNet, build_cofnet
from data.coco_dataset import COCODataset, collate_fn
from utils.nms import postprocess_detections
from utils.coco_eval import COCOEvaluator, format_metrics
from utils.box_ops import box_cxcywh_to_xyxy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate COFNet")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--checkpoint", "-w",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, or cuda:N)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.05,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.5,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=300,
        help="Maximum detections per image",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_output",
        help="Output directory for visualizations and results",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and merge config files."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle base config inheritance
    if '__base__' in config:
        base_path = Path(config_path).parent / config['__base__']
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        # Merge: config overrides base
        base_config = deep_merge(base_config, config)
        del base_config['__base__']
        config = base_config

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def build_val_dataloader(config: dict, split: str = 'val') -> torch.utils.data.DataLoader:
    """Build validation/test dataloader."""
    data_config = config.get('data', {})

    image_size = tuple(data_config.get('image_size', [640, 640]))

    if split == 'val':
        img_folder = data_config.get('val_img_folder', data_config.get('train_img_folder'))
        ann_file = data_config.get('val_ann_file', data_config.get('train_ann_file'))
    else:
        img_folder = data_config.get('test_img_folder', data_config.get('val_img_folder'))
        ann_file = data_config.get('test_ann_file', data_config.get('val_ann_file'))

    dataset = COCODataset(
        img_folder=img_folder,
        ann_file=ann_file,
        image_size=image_size,
        is_train=False,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Use batch_size=1 for accurate evaluation
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return loader


def visualize_predictions(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    output_path: str,
    gt_boxes: np.ndarray = None,
    score_threshold: float = 0.3,
):
    """
    Draw predictions on image and save.

    Args:
        image: [H, W, 3] RGB image
        boxes: [N, 4] boxes in (x1, y1, x2, y2) format
        scores: [N] confidence scores
        labels: [N] class labels
        class_names: List of class names
        output_path: Path to save visualization
        gt_boxes: Optional GT boxes to draw
        score_threshold: Only draw boxes above this score
    """
    # Convert to BGR for cv2
    vis = image.copy()
    if vis.max() <= 1.0:
        vis = (vis * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    # Color palette
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    # Draw GT boxes if provided
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box in gt_boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (128, 128, 128), 2)

    # Draw predictions
    for i in range(len(boxes)):
        if scores[i] < score_threshold:
            continue

        box = boxes[i].astype(int)
        score = scores[i]
        label = int(labels[i])
        color = colors[label % len(colors)]

        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label text
        class_name = class_names[label] if label < len(class_names) else f"class_{label}"
        text = f"{class_name}: {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(vis, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
        cv2.putText(vis, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(output_path, vis)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    conf_threshold: float = 0.05,
    nms_threshold: float = 0.5,
    max_detections: int = 300,
    visualize: bool = False,
    output_dir: str = "./eval_output",
    class_names: list = None,
) -> dict:
    """
    Run evaluation on dataset.

    Returns:
        Dict with evaluation metrics
    """
    model.eval()

    evaluator = COCOEvaluator()
    output_path = Path(output_dir)
    if visualize:
        vis_dir = output_path / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    all_ground_truths = []

    total_time = 0
    num_images = 0

    pbar = tqdm(dataloader, desc="Evaluating")
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        targets = batch['targets']

        # Time inference
        start_time = time.time()
        outputs = model(images)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        inference_time = time.time() - start_time

        total_time += inference_time
        num_images += images.shape[0]

        # Post-process predictions
        pred_boxes = outputs['pred_boxes']
        pred_logits = outputs['pred_logits']

        results = postprocess_detections(
            pred_boxes,
            pred_logits,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            max_detections=max_detections,
        )

        # Collect for evaluation
        for b, result in enumerate(results):
            # Convert GT boxes from normalized cxcywh to xyxy
            gt_boxes = targets[b]['boxes']
            gt_labels = targets[b]['labels']

            if len(gt_boxes) > 0:
                # Denormalize
                h, w = images.shape[-2:]
                gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)
                gt_boxes_xyxy[:, [0, 2]] *= w
                gt_boxes_xyxy[:, [1, 3]] *= h
            else:
                gt_boxes_xyxy = gt_boxes

            # Denormalize predictions
            pred_boxes_denorm = result['boxes'].clone()
            if len(pred_boxes_denorm) > 0:
                pred_boxes_denorm[:, [0, 2]] *= w
                pred_boxes_denorm[:, [1, 3]] *= h

            all_predictions.append({
                'boxes': pred_boxes_denorm,
                'scores': result['scores'],
                'labels': result['labels'],
            })

            all_ground_truths.append({
                'boxes': gt_boxes_xyxy,
                'labels': gt_labels,
            })

            # Visualize if requested
            if visualize and batch_idx < 100:  # Limit visualizations
                image_np = images[b].cpu().numpy().transpose(1, 2, 0)
                # Denormalize image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)

                vis_path = vis_dir / f"{batch_idx:06d}.jpg"
                visualize_predictions(
                    image_np,
                    pred_boxes_denorm.cpu().numpy(),
                    result['scores'].cpu().numpy(),
                    result['labels'].cpu().numpy(),
                    class_names or [],
                    str(vis_path),
                    gt_boxes_xyxy.cpu().numpy() if len(gt_boxes_xyxy) > 0 else None,
                )

        # Update progress bar
        avg_time = total_time / num_images
        pbar.set_postfix({'FPS': f'{1/avg_time:.1f}'})

    # Compute metrics
    evaluator.update(all_predictions, all_ground_truths)
    metrics = evaluator.compute()

    # Add timing info
    metrics['avg_inference_time'] = total_time / num_images
    metrics['fps'] = num_images / total_time

    return metrics


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    print("=" * 60)
    print("COFNet Evaluation")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    print("=" * 60)

    # Build model
    model_config = config.get('model', {})
    model = build_cofnet(model_config)

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build dataloader
    print("\nBuilding dataloader...")
    dataloader = build_val_dataloader(config, args.split)
    print(f"  Images: {len(dataloader.dataset)}")

    # Get class names
    class_names = dataloader.dataset.get_category_names()
    print(f"  Classes: {class_names}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    print("\n" + "=" * 60)
    print("Running evaluation...")
    print("=" * 60)

    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        max_detections=args.max_detections,
        visualize=args.visualize,
        output_dir=args.output_dir,
        class_names=class_names,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(format_metrics(metrics))
    print(f"\nFPS: {metrics['fps']:.1f}")
    print(f"Avg inference time: {metrics['avg_inference_time']*1000:.1f} ms")

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        # Convert numpy/tensor types to python types
        results_dict = {}
        for k, v in metrics.items():
            if k == 'per_class':
                results_dict[k] = {str(cls): {kk: float(vv) for kk, vv in clsv.items()}
                                   for cls, clsv in v.items()}
            elif isinstance(v, (np.floating, np.integer)):
                results_dict[k] = float(v)
            else:
                results_dict[k] = v
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if args.visualize:
        print(f"Visualizations saved to {output_dir}/visualizations/")


if __name__ == "__main__":
    main()
