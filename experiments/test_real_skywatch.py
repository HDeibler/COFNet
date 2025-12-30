#!/usr/bin/env python3
"""
Test compression-based detection on REAL SkyWatch dataset.

This tests the core thesis: Objects are regions of LOW complexity
against a HIGH complexity background.

Classes:
- 0: Plane
- 1: WildLife
- 2: meteorite
"""

import os
import sys
import gzip
import numpy as np
from pathlib import Path

# Check dependencies
try:
    from PIL import Image
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow", "-q"])
    from PIL import Image


DATASET_PATH = Path("/home/user/COFNet/Skywatch4-2")
CLASS_NAMES = ["Plane", "WildLife", "meteorite"]


def load_yolo_labels(label_path):
    """Load YOLO format labels: class x_center y_center width height"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])
                boxes.append({
                    'class': cls,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': w,
                    'height': h
                })
    return boxes


def yolo_to_pixel(box, img_width, img_height):
    """Convert YOLO normalized coords to pixel coords."""
    x_center = box['x_center'] * img_width
    y_center = box['y_center'] * img_height
    w = box['width'] * img_width
    h = box['height'] * img_height

    x1 = int(x_center - w/2)
    y1 = int(y_center - h/2)
    x2 = int(x_center + w/2)
    y2 = int(y_center + h/2)

    return max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)


def compute_patch_complexity(img_array, x1, y1, x2, y2):
    """Compute compression ratio for a patch."""
    if x2 <= x1 or y2 <= y1:
        return 1.0

    patch = img_array[y1:y2, x1:x2]
    if patch.size == 0:
        return 1.0

    raw = patch.tobytes()
    if len(raw) == 0:
        return 1.0

    compressed = gzip.compress(raw, compresslevel=9)
    return len(compressed) / len(raw)


def compute_background_complexity(img_array, boxes):
    """Compute complexity of regions NOT containing objects."""
    h, w = img_array.shape[:2]

    # Create mask of object regions
    mask = np.zeros((h, w), dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = yolo_to_pixel(box, w, h)
        mask[y1:y2, x1:x2] = True

    # Sample random background patches
    complexities = []
    patch_size = 32

    for _ in range(20):  # Sample 20 background patches
        x = np.random.randint(0, max(1, w - patch_size))
        y = np.random.randint(0, max(1, h - patch_size))

        # Check if patch overlaps with any object
        patch_mask = mask[y:y+patch_size, x:x+patch_size]
        if patch_mask.sum() < patch_size * patch_size * 0.1:  # <10% overlap
            c = compute_patch_complexity(img_array, x, y, x+patch_size, y+patch_size)
            complexities.append(c)

    return np.mean(complexities) if complexities else 0.5


def analyze_image(img_path, label_path):
    """Analyze compression complexity of objects vs background."""
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    boxes = load_yolo_labels(label_path)
    if not boxes:
        return None  # Skip images without objects

    # Compute object complexities
    object_complexities = []
    for box in boxes:
        x1, y1, x2, y2 = yolo_to_pixel(box, w, h)
        c = compute_patch_complexity(img_array, x1, y1, x2, y2)
        object_complexities.append({
            'class': box['class'],
            'complexity': c,
            'size': (x2-x1) * (y2-y1)
        })

    # Compute background complexity
    bg_complexity = compute_background_complexity(img_array, boxes)

    return {
        'objects': object_complexities,
        'background': bg_complexity,
        'n_objects': len(boxes)
    }


def sliding_window_detection(img_array, window_sizes=[32, 64, 96], stride=16, threshold=0.3):
    """Detect objects via compression - find low complexity regions."""
    h, w = img_array.shape[:2]
    detections = []

    for ws in window_sizes:
        if ws > min(h, w):
            continue

        for y in range(0, h - ws, stride):
            for x in range(0, w - ws, stride):
                c = compute_patch_complexity(img_array, x, y, x+ws, y+ws)
                if c < threshold:  # Low complexity = potential object
                    detections.append({
                        'x1': x, 'y1': y,
                        'x2': x+ws, 'y2': y+ws,
                        'complexity': c,
                        'confidence': 1.0 - c  # Lower complexity = higher confidence
                    })

    return detections


def iou(box1, box2):
    """Compute IoU between two boxes."""
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def nms(detections, iou_threshold=0.5):
    """Non-maximum suppression."""
    if not detections:
        return []

    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [d for d in detections if iou(best, d) < iou_threshold]

    return kept


def evaluate_detection(img_path, label_path, complexity_threshold=0.35):
    """Evaluate compression-based detection against ground truth."""
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    gt_boxes = load_yolo_labels(label_path)
    if not gt_boxes:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0}

    # Convert GT to pixel coords
    gt_pixel = []
    for box in gt_boxes:
        x1, y1, x2, y2 = yolo_to_pixel(box, w, h)
        gt_pixel.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': box['class']})

    # Run compression detection
    detections = sliding_window_detection(img_array, threshold=complexity_threshold)
    detections = nms(detections)

    # Match detections to GT
    matched_gt = set()
    tp = 0
    for det in detections:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(gt_pixel):
            if i not in matched_gt:
                iou_val = iou(det, gt)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = i

        if best_iou >= 0.3:  # Loose IoU threshold
            tp += 1
            matched_gt.add(best_gt_idx)

    fp = len(detections) - tp
    fn = len(gt_boxes) - tp

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'gt': len(gt_boxes),
        'n_detections': len(detections)
    }


def main():
    print("=" * 60)
    print("COMPRESSION-BASED DETECTION ON REAL SKYWATCH DATA")
    print("=" * 60)
    print("\nThesis: Objects have LOWER compression complexity than background")
    print("Classes: Plane, WildLife, meteorite\n")

    # Phase 1: Analyze complexity patterns
    print("-" * 60)
    print("PHASE 1: Complexity Analysis (Object vs Background)")
    print("-" * 60)

    train_images = list((DATASET_PATH / "train" / "images").glob("*.jpg"))
    np.random.seed(42)
    sample_images = np.random.choice(train_images, min(100, len(train_images)), replace=False)

    all_object_complexity = {0: [], 1: [], 2: []}  # By class
    all_background_complexity = []

    for img_path in sample_images:
        label_path = str(img_path).replace("/images/", "/labels/").replace(".jpg", ".txt")
        result = analyze_image(str(img_path), label_path)

        if result:
            for obj in result['objects']:
                all_object_complexity[obj['class']].append(obj['complexity'])
            all_background_complexity.append(result['background'])

    print("\nComplexity Statistics (compression ratio, lower = more compressible):")
    print(f"  Background: {np.mean(all_background_complexity):.3f} ± {np.std(all_background_complexity):.3f}")

    for cls_id, name in enumerate(CLASS_NAMES):
        if all_object_complexity[cls_id]:
            mean_c = np.mean(all_object_complexity[cls_id])
            std_c = np.std(all_object_complexity[cls_id])
            n = len(all_object_complexity[cls_id])
            print(f"  {name:10s}: {mean_c:.3f} ± {std_c:.3f} (n={n})")

    # Check if thesis holds
    all_objects = []
    for v in all_object_complexity.values():
        all_objects.extend(v)

    bg_mean = np.mean(all_background_complexity)
    obj_mean = np.mean(all_objects)

    print(f"\n  Overall Object Mean: {obj_mean:.3f}")
    print(f"  Overall Background Mean: {bg_mean:.3f}")

    if obj_mean < bg_mean:
        print("\n  ✓ THESIS CONFIRMED: Objects are more compressible than background!")
        print(f"    Difference: {bg_mean - obj_mean:.3f} ({(bg_mean - obj_mean)/bg_mean*100:.1f}% more compressible)")
    else:
        print("\n  ✗ THESIS NOT CONFIRMED on this dataset")
        print(f"    Objects are {obj_mean - bg_mean:.3f} LESS compressible than background")

    # Phase 2: Detection Evaluation
    print("\n" + "-" * 60)
    print("PHASE 2: Detection Evaluation")
    print("-" * 60)

    # Test on validation set
    val_images = list((DATASET_PATH / "valid" / "images").glob("*.jpg"))
    test_images = np.random.choice(val_images, min(50, len(val_images)), replace=False)

    total_tp, total_fp, total_fn, total_gt = 0, 0, 0, 0

    print(f"\nEvaluating on {len(test_images)} validation images...")

    for i, img_path in enumerate(test_images):
        label_path = str(img_path).replace("/images/", "/labels/").replace(".jpg", ".txt")
        result = evaluate_detection(str(img_path), label_path)

        total_tp += result['tp']
        total_fp += result['fp']
        total_fn += result['fn']
        total_gt += result['gt']

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(test_images)} images...")

    # Compute metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nDetection Results:")
    print(f"  True Positives:  {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"  Ground Truth:    {total_gt}")
    print(f"\n  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
This detector uses ZERO learned parameters for detection.
It finds objects purely through compression analysis.

The core idea: If you can describe a region with a short program
(high compression), it likely contains structure (an object).
Random noise/texture has high Kolmogorov complexity.

This is a proof-of-concept. Real improvements would come from:
1. Better compression approximations (learned codecs)
2. Multi-scale analysis
3. Compression relative to learned background model
4. Hybrid: Compression proposals + learned classifier
""")


if __name__ == "__main__":
    main()
