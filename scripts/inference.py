#!/usr/bin/env python3
"""
COFNet Inference Script

Run inference on images or video with a trained COFNet model.

Usage:
    # Single image
    python scripts/inference.py --checkpoint output/best.pth --input image.jpg

    # Directory of images
    python scripts/inference.py --checkpoint output/best.pth --input images/ --output results/

    # Video
    python scripts/inference.py --checkpoint output/best.pth --input video.mp4 --output output.mp4
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.cofnet import COFNet, build_cofnet
from utils.nms import postprocess_detections
from utils.box_ops import box_cxcywh_to_xyxy


# Default class names (COCO)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush',
]

# SkyWatch classes
SKYWATCH_CLASSES = ['plane', 'wildlife', 'meteorite']


def parse_args():
    parser = argparse.ArgumentParser(description="COFNet Inference")
    parser.add_argument(
        "--checkpoint", "-w",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config YAML (optional, uses checkpoint config if available)",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input image, directory, or video path",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Output path (file or directory)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[1280, 1280],
        help="Input image size (height, width)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.5,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="Class names (or 'coco' or 'skywatch')",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display results (just save)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if '__base__' in config:
        base_path = Path(config_path).parent / config['__base__']
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        for key, value in config.items():
            if key != '__base__':
                if isinstance(value, dict) and key in base_config:
                    base_config[key].update(value)
                else:
                    base_config[key] = value
        config = base_config

    return config


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
) -> Tuple[torch.Tensor, Tuple[int, int], float]:
    """
    Preprocess image for inference.

    Returns:
        (tensor, original_size, scale)
    """
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = target_size

    # Resize maintaining aspect ratio
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Pad to target size
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    # Normalize
    tensor = padded.astype(np.float32) / 255.0
    tensor = (tensor - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0)

    return tensor, (orig_h, orig_w), scale


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
) -> np.ndarray:
    """Draw detection boxes on image."""
    vis = image.copy()

    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Purple
        (255, 128, 0),  # Orange
    ]

    for i in range(len(boxes)):
        box = boxes[i].astype(int)
        score = scores[i]
        label = int(labels[i])
        color = colors[label % len(colors)]

        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label
        class_name = class_names[label] if label < len(class_names) else f"class_{label}"
        text = f"{class_name}: {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        cv2.rectangle(vis, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(vis, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    target_size: Tuple[int, int],
    conf_threshold: float,
    nms_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on a single image.

    Returns:
        (boxes, scores, labels) in original image coordinates
    """
    # Preprocess
    tensor, (orig_h, orig_w), scale = preprocess_image(image, target_size)
    tensor = tensor.to(device)

    # Forward pass
    outputs = model(tensor)

    # Post-process
    results = postprocess_detections(
        outputs['pred_boxes'],
        outputs['pred_logits'],
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )[0]

    boxes = results['boxes'].cpu().numpy()
    scores = results['scores'].cpu().numpy()
    labels = results['labels'].cpu().numpy()

    # Scale boxes back to original image size
    if len(boxes) > 0:
        # boxes are already in xyxy format from postprocess
        # but they're at target_size, need to scale back
        target_h, target_w = target_size
        boxes[:, [0, 2]] *= target_w
        boxes[:, [1, 3]] *= target_h

        # Unpad and unscale
        boxes[:, [0, 2]] /= scale
        boxes[:, [1, 3]] /= scale

        # Clip to image bounds
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    return boxes, scores, labels


def process_image(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    args,
    class_names: List[str],
) -> np.ndarray:
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes, scores, labels = run_inference(
        model, image_rgb, device,
        tuple(args.image_size),
        args.conf_threshold,
        args.nms_threshold,
    )

    vis = draw_detections(image, boxes, scores, labels, class_names)

    return vis


def process_video(
    model: torch.nn.Module,
    video_path: str,
    output_path: str,
    device: torch.device,
    args,
    class_names: List[str],
):
    """Process a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {total_frames} frames @ {fps} FPS")

    frame_idx = 0
    total_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes, scores, labels = run_inference(
            model, frame_rgb, device,
            tuple(args.image_size),
            args.conf_threshold,
            args.nms_threshold,
        )
        total_time += time.time() - start

        vis = draw_detections(frame, boxes, scores, labels, class_names)

        # Add FPS counter
        avg_fps = (frame_idx + 1) / total_time if total_time > 0 else 0
        cv2.putText(vis, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(vis)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames ({avg_fps:.1f} FPS)")

    cap.release()
    out.release()

    print(f"Saved to {output_path}")
    print(f"Average FPS: {frame_idx / total_time:.1f}")


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Get model config
    if args.config:
        config = load_config(args.config)
        model_config = config.get('model', {})
    elif 'config' in checkpoint:
        model_config = checkpoint['config'].get('model', {})
    else:
        # Default config
        model_config = {
            'num_classes': 80,
            'backbone_dims': [96, 192, 384, 768],
            'csf_dim': 256,
            'num_queries': 300,
        }

    model = build_cofnet(model_config)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Get class names
    if args.classes:
        if args.classes[0] == 'coco':
            class_names = COCO_CLASSES
        elif args.classes[0] == 'skywatch':
            class_names = SKYWATCH_CLASSES
        else:
            class_names = args.classes
    else:
        num_classes = model_config.get('num_classes', 80)
        if num_classes == 3:
            class_names = SKYWATCH_CLASSES
        else:
            class_names = COCO_CLASSES[:num_classes]

    print(f"Classes: {class_names}")

    # Process input
    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        ext = input_path.suffix.lower()
        if ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # Video
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.is_dir():
                output_path = output_path / f"output{ext}"
            process_video(model, str(input_path), str(output_path), device, args, class_names)
        else:
            # Single image
            vis = process_image(model, str(input_path), device, args, class_names)

            if output_path.suffix:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), vis)
                print(f"Saved to {output_path}")
            else:
                output_path.mkdir(parents=True, exist_ok=True)
                out_file = output_path / input_path.name
                cv2.imwrite(str(out_file), vis)
                print(f"Saved to {out_file}")

            if not args.no_display:
                cv2.imshow("COFNet Detection", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    elif input_path.is_dir():
        # Directory of images
        output_path.mkdir(parents=True, exist_ok=True)

        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        print(f"Found {len(image_files)} images")

        for img_path in image_files:
            try:
                vis = process_image(model, str(img_path), device, args, class_names)
                out_file = output_path / img_path.name
                cv2.imwrite(str(out_file), vis)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        print(f"Saved results to {output_path}")

    else:
        raise ValueError(f"Input path not found: {input_path}")


if __name__ == "__main__":
    main()
