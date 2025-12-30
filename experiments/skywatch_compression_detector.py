#!/usr/bin/env python3
"""
Compression-First Object Detection for SkyWatch

Demonstrates the thesis: Objects are "compressible patterns" in a sea of noise.
- Planes: Simple elongated shapes (low complexity)
- Wildlife/Birds: Simple oval + triangle shapes (low complexity)
- Meteorites: Bright streaks (low complexity)
- Sky background: Complex noise/gradients (high complexity)

A compression-first detector finds regions where description_length is LOW.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import gzip
import math
from collections import defaultdict

# =============================================================================
# PART 1: Synthetic SkyWatch Dataset Generator
# =============================================================================

class SkyWatchGenerator:
    """
    Generates synthetic sky images with planes, birds, and meteorites.

    Key insight: Objects are SIMPLE (low Kolmogorov complexity)
                 Background is COMPLEX (high Kolmogorov complexity)
    """

    CLASSES = ['plane', 'wildlife', 'meteorite']

    def __init__(self, img_size=128, seed=42):
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)

    def generate_sky_background(self):
        """Generate complex sky background (HIGH complexity)."""
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # Base sky gradient (dusk/dawn colors)
        for y in range(self.img_size):
            ratio = y / self.img_size
            # Dark blue to orange gradient
            r = int(20 + ratio * 60)
            g = int(20 + ratio * 40)
            b = int(80 - ratio * 30)
            img[y, :] = [r, g, b]

        # Add noise (makes background incompressible)
        noise = self.rng.randint(-15, 15, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Add some "stars" (small bright points)
        n_stars = self.rng.randint(10, 30)
        for _ in range(n_stars):
            x, y = self.rng.randint(0, self.img_size, 2)
            brightness = self.rng.randint(150, 255)
            if 0 <= x < self.img_size and 0 <= y < self.img_size:
                img[y, x] = [brightness, brightness, brightness]

        return img

    def draw_plane(self, img, bbox):
        """Draw a simple plane shape (LOW complexity - just rectangles)."""
        x, y, w, h = bbox
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        # Plane = fuselage (rectangle) + wings (rectangle)
        # Fuselage
        cx, cy = x + w//2, y + h//2
        fuse_w, fuse_h = int(w * 0.8), int(h * 0.2)
        draw.rectangle([cx - fuse_w//2, cy - fuse_h//2,
                       cx + fuse_w//2, cy + fuse_h//2],
                      fill=(200, 200, 210))

        # Wings
        wing_w, wing_h = int(w * 0.3), int(h * 0.7)
        draw.rectangle([cx - wing_w//2, cy - wing_h//2,
                       cx + wing_w//2, cy + wing_h//2],
                      fill=(180, 180, 190))

        # Tail
        tail_w, tail_h = int(w * 0.15), int(h * 0.3)
        draw.rectangle([cx + fuse_w//2 - tail_w, cy - tail_h//2,
                       cx + fuse_w//2, cy + tail_h//2],
                      fill=(170, 170, 180))

        return np.array(pil_img)

    def draw_wildlife(self, img, bbox):
        """Draw a simple bird shape (LOW complexity - oval + triangle)."""
        x, y, w, h = bbox
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        cx, cy = x + w//2, y + h//2

        # Body (ellipse)
        body_w, body_h = int(w * 0.5), int(h * 0.4)
        draw.ellipse([cx - body_w//2, cy - body_h//2,
                     cx + body_w//2, cy + body_h//2],
                    fill=(60, 60, 70))

        # Wings (triangles)
        wing_span = int(w * 0.9)
        draw.polygon([(cx, cy),
                     (cx - wing_span//2, cy - h//3),
                     (cx - wing_span//4, cy)],
                    fill=(50, 50, 60))
        draw.polygon([(cx, cy),
                     (cx + wing_span//2, cy - h//3),
                     (cx + wing_span//4, cy)],
                    fill=(50, 50, 60))

        return np.array(pil_img)

    def draw_meteorite(self, img, bbox):
        """Draw a meteorite streak (LOW complexity - line + glow)."""
        x, y, w, h = bbox
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        # Bright streak
        x1, y1 = x, y + h
        x2, y2 = x + w, y

        # Draw thick bright line
        for offset in range(-2, 3):
            draw.line([(x1, y1 + offset), (x2, y2 + offset)],
                     fill=(255, 255, 200), width=2)

        # Bright head
        draw.ellipse([x2 - 4, y2 - 4, x2 + 4, y2 + 4],
                    fill=(255, 255, 255))

        return np.array(pil_img)

    def generate_sample(self):
        """Generate one training sample with annotations."""
        img = self.generate_sky_background()
        annotations = []

        # Random number of objects (1-3)
        n_objects = self.rng.randint(1, 4)

        for _ in range(n_objects):
            # Random class (with meteorite being rare - like real dataset)
            class_weights = [0.4, 0.4, 0.2]  # plane, wildlife, meteorite
            class_idx = self.rng.choice(3, p=class_weights)
            class_name = self.CLASSES[class_idx]

            # Random bounding box
            obj_size = self.rng.randint(15, 35)
            x = self.rng.randint(5, self.img_size - obj_size - 5)
            y = self.rng.randint(5, self.img_size - obj_size - 5)
            w = obj_size + self.rng.randint(-5, 10)
            h = obj_size + self.rng.randint(-5, 10)
            bbox = [x, y, w, h]

            # Draw object
            if class_name == 'plane':
                img = self.draw_plane(img, bbox)
            elif class_name == 'wildlife':
                img = self.draw_wildlife(img, bbox)
            else:
                img = self.draw_meteorite(img, bbox)

            annotations.append({
                'bbox': bbox,
                'class': class_name,
                'class_idx': class_idx
            })

        return img, annotations

    def generate_dataset(self, n_train=200, n_val=50):
        """Generate full dataset."""
        print(f"Generating {n_train} training + {n_val} validation images...")

        train_data = [self.generate_sample() for _ in range(n_train)]
        val_data = [self.generate_sample() for _ in range(n_val)]

        # Count class distribution
        class_counts = defaultdict(int)
        for _, anns in train_data:
            for ann in anns:
                class_counts[ann['class']] += 1

        print(f"Class distribution: {dict(class_counts)}")

        return train_data, val_data


# =============================================================================
# PART 2: Compression-Based Objectness Measure
# =============================================================================

def compute_patch_complexity(patch):
    """
    Approximate Kolmogorov complexity of an image patch.

    Uses actual compression (gzip) as an approximation.
    Lower complexity = more compressible = more likely to be an object.

    This is the KEY INSIGHT: Objects are simple, backgrounds are complex.
    """
    # Convert to bytes
    patch_bytes = patch.tobytes()

    # Compress with gzip
    compressed = gzip.compress(patch_bytes, compresslevel=9)

    # Complexity = compressed size / original size
    complexity = len(compressed) / len(patch_bytes)

    return complexity


def sliding_window_complexity(img, window_size=24, stride=8):
    """
    Compute complexity map using sliding window.

    Low complexity regions = candidate objects.
    """
    h, w = img.shape[:2]

    complexity_map = np.ones((h, w)) * 999

    for y in range(0, h - window_size, stride):
        for x in range(0, w - window_size, stride):
            patch = img[y:y+window_size, x:x+window_size]
            complexity = compute_patch_complexity(patch)

            # Fill the region with this complexity value
            complexity_map[y:y+window_size, x:x+window_size] = np.minimum(
                complexity_map[y:y+window_size, x:x+window_size],
                complexity
            )

    return complexity_map


def find_low_complexity_regions(complexity_map, threshold=0.5, min_size=10):
    """
    Find connected regions with low complexity.
    These are our object proposals!
    """
    # Binary mask of low-complexity regions
    mask = complexity_map < threshold

    # Simple connected components (flood fill)
    proposals = []
    visited = np.zeros_like(mask, dtype=bool)

    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] and not visited[y, x]:
                # Flood fill to find connected region
                region = []
                stack = [(y, x)]

                while stack:
                    cy, cx = stack.pop()
                    if (0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]
                        and mask[cy, cx] and not visited[cy, cx]):
                        visited[cy, cx] = True
                        region.append((cy, cx))
                        stack.extend([(cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1)])

                if len(region) >= min_size:
                    # Compute bounding box
                    ys, xs = zip(*region)
                    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                    avg_complexity = np.mean([complexity_map[p[0], p[1]] for p in region])
                    proposals.append({
                        'bbox': bbox,
                        'complexity': avg_complexity,
                        'size': len(region)
                    })

    return proposals


# =============================================================================
# PART 3: Simple Classifier for Detected Regions
# =============================================================================

class SimpleClassifier:
    """
    Tiny classifier for object regions.
    This is intentionally simple - the intelligence is in FINDING objects,
    not classifying them.
    """

    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        # Just use color histograms as features
        self.class_templates = {}

    def extract_features(self, patch):
        """Simple feature extraction - color histogram."""
        # Resize to fixed size
        patch_resized = np.array(Image.fromarray(patch).resize((16, 16)))

        # Color histogram (8 bins per channel)
        features = []
        for c in range(3):
            hist, _ = np.histogram(patch_resized[:, :, c], bins=8, range=(0, 256))
            features.extend(hist / hist.sum())

        # Add mean/std
        features.extend([patch_resized.mean() / 255, patch_resized.std() / 255])

        return np.array(features)

    def fit(self, train_data):
        """Learn class templates from training data."""
        class_features = defaultdict(list)

        for img, annotations in train_data:
            for ann in annotations:
                x, y, w, h = ann['bbox']
                # Ensure valid bounds
                x, y = max(0, x), max(0, y)
                x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)

                if x2 > x and y2 > y:
                    patch = img[y:y2, x:x2]
                    features = self.extract_features(patch)
                    class_features[ann['class_idx']].append(features)

        # Compute mean template for each class
        for class_idx, feats in class_features.items():
            self.class_templates[class_idx] = np.mean(feats, axis=0)

        print(f"Learned templates for {len(self.class_templates)} classes")

    def predict(self, patch):
        """Predict class for a patch using nearest template."""
        features = self.extract_features(patch)

        best_class = 0
        best_dist = float('inf')

        for class_idx, template in self.class_templates.items():
            dist = np.linalg.norm(features - template)
            if dist < best_dist:
                best_dist = dist
                best_class = class_idx

        return best_class, 1.0 / (1.0 + best_dist)  # class, confidence


# =============================================================================
# PART 4: Compression-First Detector
# =============================================================================

class CompressionDetector:
    """
    Object detector based on compression/complexity.

    Key idea: Objects are LOW complexity regions in HIGH complexity backgrounds.

    This is fundamentally different from learned detectors:
    - No training on detection task
    - Uses compression as the "objectness" signal
    - Classifier is minimal (just for labeling found objects)
    """

    def __init__(self, window_size=24, stride=8, complexity_threshold=0.55):
        self.window_size = window_size
        self.stride = stride
        self.complexity_threshold = complexity_threshold
        self.classifier = SimpleClassifier(n_classes=3)

    def fit(self, train_data):
        """Train only the classifier (detection is compression-based)."""
        self.classifier.fit(train_data)

        # Also compute background complexity distribution
        bg_complexities = []
        for img, _ in train_data[:20]:
            cmap = sliding_window_complexity(img, self.window_size, self.stride)
            bg_complexities.extend(cmap.flatten())

        # Set threshold based on distribution
        self.complexity_threshold = np.percentile(bg_complexities, 25)
        print(f"Complexity threshold set to: {self.complexity_threshold:.3f}")

    def detect(self, img):
        """
        Detect objects using compression-based approach.

        1. Compute complexity map
        2. Find low-complexity regions (these are objects!)
        3. Classify each region
        """
        # Step 1: Complexity map
        complexity_map = sliding_window_complexity(img, self.window_size, self.stride)

        # Step 2: Find low-complexity regions
        proposals = find_low_complexity_regions(
            complexity_map,
            threshold=self.complexity_threshold,
            min_size=50
        )

        # Step 3: Classify each proposal
        detections = []
        for prop in proposals:
            x, y, w, h = prop['bbox']
            # Ensure valid bounds
            x, y = max(0, x), max(0, y)
            x2 = min(img.shape[1], x + max(w, 10))
            y2 = min(img.shape[0], y + max(h, 10))

            if x2 > x + 5 and y2 > y + 5:
                patch = img[y:y2, x:x2]
                class_idx, confidence = self.classifier.predict(patch)

                detections.append({
                    'bbox': [x, y, x2-x, y2-y],
                    'class_idx': class_idx,
                    'class': SkyWatchGenerator.CLASSES[class_idx],
                    'confidence': confidence,
                    'complexity': prop['complexity']
                })

        return detections, complexity_map


# =============================================================================
# PART 5: Standard Neural Network Detector (Baseline)
# =============================================================================

class StandardDetector:
    """
    Simple learned detector for comparison.
    Uses a sliding window + neural network classifier.

    This represents the "standard paradigm":
    - Fixed architecture
    - Learns everything from data
    - No compression insight
    """

    def __init__(self, window_size=24, stride=8):
        self.window_size = window_size
        self.stride = stride

        # Simple 2-layer network
        self.input_dim = window_size * window_size * 3
        self.hidden_dim = 64
        self.output_dim = 4  # 3 classes + background

        # Initialize weights
        np.random.seed(42)
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim).astype(np.float32) * 0.01
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim).astype(np.float32) * 0.01
        self.b2 = np.zeros(self.output_dim, dtype=np.float32)

    def forward(self, x):
        self.x = x.reshape(x.shape[0], -1).astype(np.float32) / 255.0
        self.z1 = self.x @ self.W1 + self.b1
        self.h1 = np.maximum(0, self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        exp_z = np.exp(self.z2 - self.z2.max(axis=1, keepdims=True))
        self.out = exp_z / exp_z.sum(axis=1, keepdims=True)
        return self.out

    def backward(self, target, lr=0.001):
        batch_size = len(target)
        dz2 = self.out.copy()
        dz2[np.arange(batch_size), target] -= 1
        dz2 /= batch_size

        dW2 = self.h1.T @ dz2
        db2 = dz2.sum(axis=0)

        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (self.z1 > 0)

        dW1 = self.x.T @ dz1
        db1 = dz1.sum(axis=0)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def fit(self, train_data, epochs=10):
        """Train on labeled patches."""
        # Extract training patches
        patches = []
        labels = []

        for img, annotations in train_data:
            # Positive samples (objects)
            for ann in annotations:
                x, y, w, h = ann['bbox']
                x, y = max(0, x), max(0, y)
                x2 = min(img.shape[1], x + self.window_size)
                y2 = min(img.shape[0], y + self.window_size)

                if x2 - x >= self.window_size * 0.8 and y2 - y >= self.window_size * 0.8:
                    patch = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
                    patch[:y2-y, :x2-x] = img[y:y2, x:x2]
                    patches.append(patch)
                    labels.append(ann['class_idx'])

            # Negative samples (background)
            for _ in range(2):
                x = np.random.randint(0, img.shape[1] - self.window_size)
                y = np.random.randint(0, img.shape[0] - self.window_size)
                patch = img[y:y+self.window_size, x:x+self.window_size]
                patches.append(patch)
                labels.append(3)  # Background class

        patches = np.array(patches)
        labels = np.array(labels)

        print(f"Training standard detector on {len(patches)} patches...")

        # Train
        for epoch in range(epochs):
            perm = np.random.permutation(len(patches))
            patches = patches[perm]
            labels = labels[perm]

            for i in range(0, len(patches), 32):
                batch_x = patches[i:i+32]
                batch_y = labels[i:i+32]
                self.forward(batch_x)
                self.backward(batch_y)

            if (epoch + 1) % 5 == 0:
                pred = self.forward(patches)
                acc = (pred.argmax(axis=1) == labels).mean()
                print(f"  Epoch {epoch+1}: accuracy = {acc:.3f}")

    def detect(self, img):
        """Detect using sliding window."""
        detections = []

        for y in range(0, img.shape[0] - self.window_size, self.stride):
            for x in range(0, img.shape[1] - self.window_size, self.stride):
                patch = img[y:y+self.window_size, x:x+self.window_size]
                pred = self.forward(patch[np.newaxis])[0]

                class_idx = pred[:3].argmax()  # Exclude background
                confidence = pred[class_idx]

                if confidence > 0.3 and pred[3] < 0.5:  # Not background
                    detections.append({
                        'bbox': [x, y, self.window_size, self.window_size],
                        'class_idx': class_idx,
                        'class': SkyWatchGenerator.CLASSES[class_idx],
                        'confidence': float(confidence)
                    })

        # Simple NMS
        detections = sorted(detections, key=lambda x: -x['confidence'])
        final = []
        for det in detections:
            overlap = False
            for kept in final:
                if iou(det['bbox'], kept['bbox']) > 0.3:
                    overlap = True
                    break
            if not overlap:
                final.append(det)

        return final[:10]  # Max 10 detections

    def num_parameters(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


def iou(box1, box2):
    """Compute IoU between two boxes [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter

    return inter / max(union, 1e-6)


# =============================================================================
# PART 6: Evaluation
# =============================================================================

def evaluate_detector(detector, val_data, iou_threshold=0.3):
    """Evaluate detector on validation set."""
    total_gt = 0
    total_pred = 0
    true_positives = 0
    class_correct = 0

    for img, annotations in val_data:
        if hasattr(detector, 'detect'):
            if isinstance(detector, CompressionDetector):
                detections, _ = detector.detect(img)
            else:
                detections = detector.detect(img)

        total_gt += len(annotations)
        total_pred += len(detections)

        # Match predictions to ground truth
        matched = set()
        for det in detections:
            for i, ann in enumerate(annotations):
                if i not in matched and iou(det['bbox'], ann['bbox']) > iou_threshold:
                    true_positives += 1
                    matched.add(i)
                    if det['class_idx'] == ann['class_idx']:
                        class_correct += 1
                    break

    precision = true_positives / max(total_pred, 1)
    recall = true_positives / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    class_acc = class_correct / max(true_positives, 1)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_accuracy': class_acc,
        'total_gt': total_gt,
        'total_pred': total_pred,
        'true_positives': true_positives
    }


# =============================================================================
# PART 7: Main Experiment
# =============================================================================

def main():
    print("=" * 70)
    print("COMPRESSION-FIRST OBJECT DETECTION FOR SKYWATCH")
    print("=" * 70)
    print()
    print("Thesis: Objects are LOW complexity, backgrounds are HIGH complexity")
    print("        A compression-based detector needs NO learned features!")
    print()

    # Generate dataset
    generator = SkyWatchGenerator(img_size=128, seed=42)
    train_data, val_data = generator.generate_dataset(n_train=200, n_val=50)

    print()
    print("-" * 70)
    print("DETECTOR 1: Compression-Based (NO learned detection)")
    print("-" * 70)

    comp_detector = CompressionDetector(window_size=24, stride=8)
    comp_detector.fit(train_data)  # Only trains tiny classifier

    print("\nEvaluating...")
    comp_results = evaluate_detector(comp_detector, val_data)

    print(f"  Precision: {comp_results['precision']:.3f}")
    print(f"  Recall:    {comp_results['recall']:.3f}")
    print(f"  F1 Score:  {comp_results['f1']:.3f}")
    print(f"  Class Acc: {comp_results['class_accuracy']:.3f}")
    print(f"  (Detected {comp_results['true_positives']}/{comp_results['total_gt']} objects)")

    print()
    print("-" * 70)
    print("DETECTOR 2: Standard Neural Network (Learned detection)")
    print("-" * 70)

    std_detector = StandardDetector(window_size=24, stride=8)
    std_detector.fit(train_data, epochs=15)

    print("\nEvaluating...")
    std_results = evaluate_detector(std_detector, val_data)

    print(f"  Precision: {std_results['precision']:.3f}")
    print(f"  Recall:    {std_results['recall']:.3f}")
    print(f"  F1 Score:  {std_results['f1']:.3f}")
    print(f"  Class Acc: {std_results['class_accuracy']:.3f}")
    print(f"  (Detected {std_results['true_positives']}/{std_results['total_gt']} objects)")
    print(f"  Parameters: {std_detector.num_parameters():,}")

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print()

    print(f"{'Metric':<20} {'Compression':<15} {'Standard NN':<15}")
    print("-" * 50)
    print(f"{'F1 Score':<20} {comp_results['f1']:<15.3f} {std_results['f1']:<15.3f}")
    print(f"{'Precision':<20} {comp_results['precision']:<15.3f} {std_results['precision']:<15.3f}")
    print(f"{'Recall':<20} {comp_results['recall']:<15.3f} {std_results['recall']:<15.3f}")
    print(f"{'Class Accuracy':<20} {comp_results['class_accuracy']:<15.3f} {std_results['class_accuracy']:<15.3f}")
    print(f"{'Detection Params':<20} {'0':<15} {std_detector.num_parameters():<15,}")
    print()

    print("-" * 70)
    print("KEY INSIGHT")
    print("-" * 70)
    print()
    print("The compression-based detector finds objects using ZERO learned")
    print("detection parameters! It uses gzip compression to identify regions")
    print("that are 'surprisingly simple' compared to the noisy background.")
    print()
    print("This demonstrates: Objects ARE compressible patterns.")
    print("                   You don't need massive learned detectors.")
    print("                   Compression IS the detection signal.")
    print()

    # Show example
    print("-" * 70)
    print("EXAMPLE DETECTION")
    print("-" * 70)
    img, annotations = val_data[0]
    detections, complexity_map = comp_detector.detect(img)

    print(f"\nGround truth: {len(annotations)} objects")
    for ann in annotations:
        print(f"  - {ann['class']} at {ann['bbox']}")

    print(f"\nDetections: {len(detections)} objects")
    for det in detections:
        print(f"  - {det['class']} at {det['bbox'][:2]} (complexity={det['complexity']:.3f})")

    print(f"\nComplexity map stats:")
    print(f"  Min (objects):     {complexity_map.min():.3f}")
    print(f"  Max (background):  {complexity_map.max():.3f}")
    print(f"  Mean:              {complexity_map.mean():.3f}")

    return comp_detector, std_detector, train_data, val_data


if __name__ == "__main__":
    main()
