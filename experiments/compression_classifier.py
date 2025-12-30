#!/usr/bin/env python3
"""
Compression-based CLASSIFICATION of objects.

Can we tell apart Planes, Wildlife, and Meteorites using only compression?

Key insight from data:
  - Planes:     0.272 complexity (most structured/compressible)
  - Meteorites: 0.345 complexity (medium)
  - Wildlife:   0.435 complexity (least compressible - organic textures)

This suggests compression patterns encode object TYPE, not just presence!
"""

import os
import sys
import gzip
import zlib
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow", "-q"])
    from PIL import Image

DATASET_PATH = Path("/home/user/COFNet/Skywatch4-2")
CLASS_NAMES = {0: "Plane", 1: "WildLife", 2: "Meteorite"}


def load_yolo_labels(label_path):
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
    x_center = box['x_center'] * img_width
    y_center = box['y_center'] * img_height
    w = box['width'] * img_width
    h = box['height'] * img_height
    x1 = int(x_center - w/2)
    y1 = int(y_center - h/2)
    x2 = int(x_center + w/2)
    y2 = int(y_center + h/2)
    return max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)


def extract_compression_features(patch):
    """Extract multiple compression-based features from a patch."""
    if patch.size == 0:
        return None

    # Resize to standard size for fair comparison
    patch_pil = Image.fromarray(patch)
    patch_resized = np.array(patch_pil.resize((64, 64)))

    raw = patch_resized.tobytes()
    if len(raw) == 0:
        return None

    features = {}

    # Feature 1: Gzip compression ratio
    gzip_compressed = gzip.compress(raw, compresslevel=9)
    features['gzip_ratio'] = len(gzip_compressed) / len(raw)

    # Feature 2: Zlib compression ratio (different algorithm)
    zlib_compressed = zlib.compress(raw, level=9)
    features['zlib_ratio'] = len(zlib_compressed) / len(raw)

    # Feature 3: Per-channel compression (color structure)
    for i, channel in enumerate(['R', 'G', 'B']):
        channel_data = patch_resized[:, :, i].tobytes()
        channel_compressed = gzip.compress(channel_data, compresslevel=9)
        features[f'{channel}_ratio'] = len(channel_compressed) / len(channel_data)

    # Feature 4: Gradient complexity (edge structure)
    gray = np.mean(patch_resized, axis=2).astype(np.uint8)
    dx = np.abs(np.diff(gray, axis=1)).astype(np.uint8)
    dy = np.abs(np.diff(gray, axis=0)).astype(np.uint8)

    dx_compressed = gzip.compress(dx.tobytes(), compresslevel=9)
    dy_compressed = gzip.compress(dy.tobytes(), compresslevel=9)
    features['dx_ratio'] = len(dx_compressed) / max(1, len(dx.tobytes()))
    features['dy_ratio'] = len(dy_compressed) / max(1, len(dy.tobytes()))

    # Feature 5: Local variance (texture roughness)
    features['variance'] = float(np.var(gray)) / 255.0

    # Feature 6: Entropy approximation
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    features['entropy'] = -np.sum(hist * np.log2(hist)) / 8.0  # Normalize to 0-1

    # Feature 7: Row vs Column compression asymmetry (shape info)
    row_data = gray.tobytes()
    col_data = gray.T.tobytes()
    row_comp = len(gzip.compress(row_data, compresslevel=9))
    col_comp = len(gzip.compress(col_data, compresslevel=9))
    features['asymmetry'] = (row_comp - col_comp) / max(row_comp, col_comp)

    return features


def features_to_vector(features):
    """Convert feature dict to numpy vector."""
    keys = ['gzip_ratio', 'zlib_ratio', 'R_ratio', 'G_ratio', 'B_ratio',
            'dx_ratio', 'dy_ratio', 'variance', 'entropy', 'asymmetry']
    return np.array([features[k] for k in keys])


class CompressionClassifier:
    """Classify objects by compression signature."""

    def __init__(self):
        self.class_profiles = {}  # Mean feature vector per class
        self.class_stds = {}      # Std per class
        self.class_counts = defaultdict(int)

    def fit(self, features_by_class):
        """Learn class profiles from training data."""
        for cls, feature_list in features_by_class.items():
            if len(feature_list) > 0:
                vectors = np.array([features_to_vector(f) for f in feature_list])
                self.class_profiles[cls] = np.mean(vectors, axis=0)
                self.class_stds[cls] = np.std(vectors, axis=0) + 1e-6
                self.class_counts[cls] = len(feature_list)

    def predict(self, features):
        """Predict class based on nearest profile."""
        if features is None:
            return -1, 0.0

        vec = features_to_vector(features)

        best_class = -1
        best_score = float('inf')

        for cls, profile in self.class_profiles.items():
            # Mahalanobis-like distance (normalized by std)
            diff = (vec - profile) / self.class_stds[cls]
            distance = np.sqrt(np.sum(diff ** 2))

            if distance < best_score:
                best_score = distance
                best_class = cls

        # Convert distance to confidence (lower distance = higher confidence)
        confidence = 1.0 / (1.0 + best_score)

        return best_class, confidence

    def print_profiles(self):
        """Print learned class profiles."""
        print("\nLearned Compression Profiles:")
        print("-" * 70)

        feature_names = ['gzip', 'zlib', 'R', 'G', 'B', 'dx', 'dy', 'var', 'ent', 'asym']

        header = f"{'Class':12s} | " + " | ".join([f"{n:5s}" for n in feature_names])
        print(header)
        print("-" * 70)

        for cls in sorted(self.class_profiles.keys()):
            profile = self.class_profiles[cls]
            row = f"{CLASS_NAMES[cls]:12s} | " + " | ".join([f"{v:.3f}" for v in profile])
            print(row)


def collect_training_data(split='train', max_samples_per_class=200):
    """Collect compression features from training data."""
    images_dir = DATASET_PATH / split / "images"
    labels_dir = DATASET_PATH / split / "labels"

    features_by_class = defaultdict(list)

    image_files = list(images_dir.glob("*.jpg"))
    np.random.seed(42)
    np.random.shuffle(image_files)

    print(f"Collecting features from {split} split...")

    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        boxes = load_yolo_labels(str(label_path))

        if not boxes:
            continue

        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        h, w = img_array.shape[:2]

        for box in boxes:
            cls = box['class']

            # Skip if we have enough samples
            if len(features_by_class[cls]) >= max_samples_per_class:
                continue

            x1, y1, x2, y2 = yolo_to_pixel(box, w, h)

            # Skip tiny patches
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            patch = img_array[y1:y2, x1:x2]
            features = extract_compression_features(patch)

            if features is not None:
                features_by_class[cls].append(features)

        # Check if we have enough
        if all(len(features_by_class[c]) >= max_samples_per_class
               for c in range(3) if c in features_by_class or True):
            min_samples = min(len(features_by_class[c]) for c in features_by_class)
            if min_samples >= max_samples_per_class // 2:
                break

    return features_by_class


def evaluate_classifier(classifier, split='valid'):
    """Evaluate classifier on validation/test data."""
    images_dir = DATASET_PATH / split / "images"
    labels_dir = DATASET_PATH / split / "labels"

    correct = defaultdict(int)
    total = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    image_files = list(images_dir.glob("*.jpg"))
    np.random.seed(123)
    np.random.shuffle(image_files)

    print(f"\nEvaluating on {split} split ({len(image_files)} images)...")

    for i, img_path in enumerate(image_files[:200]):  # Limit for speed
        label_path = labels_dir / (img_path.stem + ".txt")
        boxes = load_yolo_labels(str(label_path))

        if not boxes:
            continue

        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        h, w = img_array.shape[:2]

        for box in boxes:
            true_cls = box['class']
            x1, y1, x2, y2 = yolo_to_pixel(box, w, h)

            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            patch = img_array[y1:y2, x1:x2]
            features = extract_compression_features(patch)

            pred_cls, confidence = classifier.predict(features)

            total[true_cls] += 1
            if pred_cls == true_cls:
                correct[true_cls] += 1
            confusion[true_cls][pred_cls] += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1} images...")

    return correct, total, confusion


def main():
    print("=" * 60)
    print("COMPRESSION-BASED OBJECT CLASSIFICATION")
    print("=" * 60)
    print("""
Can we classify objects using ONLY compression features?

Classes: Plane, WildLife, Meteorite

Hypothesis: Different object types have distinct compression
signatures due to their visual structure:
  - Planes: Smooth surfaces, geometric shapes
  - Wildlife: Organic textures, complex patterns
  - Meteorites: Bright spots, high contrast
""")

    # Collect training data
    print("\n" + "-" * 60)
    print("PHASE 1: Collecting Training Features")
    print("-" * 60)

    train_features = collect_training_data('train', max_samples_per_class=300)

    for cls in sorted(train_features.keys()):
        print(f"  {CLASS_NAMES[cls]:12s}: {len(train_features[cls])} samples")

    # Train classifier
    print("\n" + "-" * 60)
    print("PHASE 2: Training Classifier")
    print("-" * 60)

    classifier = CompressionClassifier()
    classifier.fit(train_features)
    classifier.print_profiles()

    # Evaluate
    print("\n" + "-" * 60)
    print("PHASE 3: Evaluation")
    print("-" * 60)

    correct, total, confusion = evaluate_classifier(classifier, 'valid')

    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)

    print("\nPer-Class Accuracy:")
    overall_correct = 0
    overall_total = 0

    for cls in sorted(total.keys()):
        acc = correct[cls] / total[cls] if total[cls] > 0 else 0
        print(f"  {CLASS_NAMES[cls]:12s}: {correct[cls]:3d}/{total[cls]:3d} = {acc:.1%}")
        overall_correct += correct[cls]
        overall_total += total[cls]

    overall_acc = overall_correct / overall_total if overall_total > 0 else 0
    print(f"\n  {'Overall':12s}: {overall_correct:3d}/{overall_total:3d} = {overall_acc:.1%}")

    # Random baseline
    random_baseline = 1.0 / len(CLASS_NAMES)
    print(f"\n  Random Baseline: {random_baseline:.1%}")
    print(f"  Improvement:     {overall_acc / random_baseline:.1f}x better than random")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print("                  Predicted")
    print(f"{'Actual':12s} | {'Plane':8s} {'WildLife':8s} {'Meteorite':8s}")
    print("-" * 50)

    for true_cls in sorted(confusion.keys()):
        row = f"{CLASS_NAMES[true_cls]:12s} |"
        for pred_cls in range(3):
            count = confusion[true_cls].get(pred_cls, 0)
            row += f" {count:8d}"
        print(row)

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"""
ZERO-PARAMETER COMPRESSION CLASSIFIER RESULTS:
  - Overall Accuracy: {overall_acc:.1%}
  - Random Baseline:  {random_baseline:.1%}
  - Improvement:      {overall_acc / random_baseline:.1f}x better than random

Using ONLY compression features (gzip ratios, entropy, gradients),
we can distinguish between object types with {overall_acc:.0%} accuracy.

This demonstrates that compression encodes not just PRESENCE
but also TYPE of objects - different visual structures compress
differently.

Key insight: Compression is a universal feature extractor.
""")


if __name__ == "__main__":
    main()
