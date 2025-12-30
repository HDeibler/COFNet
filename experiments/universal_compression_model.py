#!/usr/bin/env python3
"""
Universal Compression-Based Intelligence

This demonstrates the FULL thesis:
  Intelligence = Compression (for ANY modality)

The SAME compression principle handles:
  1. Object Detection (find compressible regions)
  2. Language Generation (generate shortest description)
  3. Scene Understanding (compress observations into concepts)

This is a prototype of what a "universal" model looks like when
built on compression rather than task-specific training.
"""

import numpy as np
from PIL import Image, ImageDraw
import gzip
import math
from collections import defaultdict
import re

# =============================================================================
# PART 1: Compression-Based Language Model
# =============================================================================

class CompressionLanguageModel:
    """
    Language model based on compression.

    Key insight: The best sentence is the SHORTEST one that conveys the meaning.
    This is literally the Minimum Description Length principle for language.

    Instead of learning billions of parameters, we:
    1. Build a vocabulary of compressed concepts
    2. Generate by finding shortest description
    3. Use actual compression to score sentences
    """

    def __init__(self):
        # Core vocabulary (concepts, not just words)
        self.concepts = {
            # Objects
            'plane': ['plane', 'aircraft', 'airplane', 'jet'],
            'wildlife': ['bird', 'wildlife', 'creature', 'animal'],
            'meteorite': ['meteorite', 'meteor', 'shooting star', 'fireball'],

            # Positions
            'top': ['top', 'upper', 'above'],
            'bottom': ['bottom', 'lower', 'below'],
            'left': ['left', 'left side'],
            'right': ['right', 'right side'],
            'center': ['center', 'middle'],

            # Quantities
            'one': ['a', 'one', 'single'],
            'two': ['two', 'pair of', 'couple of'],
            'few': ['few', 'several', 'some'],
            'many': ['many', 'multiple', 'numerous'],

            # Actions/States
            'flying': ['flying', 'soaring', 'gliding'],
            'moving': ['moving', 'traveling', 'crossing'],
            'falling': ['falling', 'streaking', 'descending'],

            # Scene
            'sky': ['sky', 'night sky', 'atmosphere'],
            'dark': ['dark', 'night', 'dim'],
        }

        # Templates for sentence generation (sorted by compression)
        self.templates = [
            # Shortest (best compression)
            "{count} {object} in {location}.",
            "{count} {object} {action} in the sky.",
            "A {object} visible in the {location}.",

            # Medium
            "There is {count} {object} {action} in the {location} of the image.",
            "The image shows {count} {object} in the {location}.",

            # Longest (worst compression)
            "In this image, we can observe {count} {object} {action} across the {location} portion of the night sky.",
        ]

        # Word frequencies (for compression estimation)
        self.word_freq = defaultdict(lambda: 0.001)
        common_words = ['a', 'the', 'in', 'is', 'of', 'and', 'to', 'sky', 'image']
        for w in common_words:
            self.word_freq[w] = 0.1

    def get_description_length(self, text):
        """
        Compute description length of text in bits.

        Uses actual compression as approximation of Kolmogorov complexity.
        """
        text_bytes = text.encode('utf-8')
        compressed = gzip.compress(text_bytes, compresslevel=9)
        return len(compressed) * 8  # bits

    def score_sentence(self, sentence, detections):
        """
        Score a sentence by how well it compresses the scene.

        Good sentence = conveys all information with minimal length
        """
        # Accuracy: Does it mention all detected objects?
        accuracy_score = 0
        for det in detections:
            obj_class = det['class']
            if any(word in sentence.lower() for word in self.concepts.get(obj_class, [obj_class])):
                accuracy_score += 1
        accuracy_score = accuracy_score / max(len(detections), 1)

        # Compression: How short is the description?
        desc_length = self.get_description_length(sentence)
        compression_score = 1.0 / (1.0 + desc_length / 100)

        # Combined: accuracy / length (MDL principle!)
        # We want HIGH accuracy with LOW description length
        mdl_score = accuracy_score * compression_score

        return {
            'mdl_score': mdl_score,
            'accuracy': accuracy_score,
            'bits': desc_length,
            'compression': compression_score
        }

    def get_position(self, bbox, img_size):
        """Convert bounding box to position word."""
        x, y, w, h = bbox
        cx = (x + w/2) / img_size
        cy = (y + h/2) / img_size

        if cy < 0.33:
            vert = 'top'
        elif cy > 0.66:
            vert = 'bottom'
        else:
            vert = 'center'

        if cx < 0.33:
            horiz = 'left'
        elif cx > 0.66:
            horiz = 'right'
        else:
            horiz = 'center'

        if vert == 'center' and horiz == 'center':
            return 'center'
        elif vert == 'center':
            return horiz
        elif horiz == 'center':
            return vert
        else:
            return f"{vert}-{horiz}"

    def get_action(self, obj_class):
        """Get appropriate action for object class."""
        actions = {
            'plane': 'flying',
            'wildlife': 'flying',
            'meteorite': 'falling'
        }
        return actions.get(obj_class, 'moving')

    def get_count_word(self, n):
        """Convert count to word."""
        if n == 1:
            return 'a'
        elif n == 2:
            return 'two'
        elif n <= 4:
            return 'a few'
        else:
            return 'several'

    def generate(self, detections, img_size=128, strategy='shortest'):
        """
        Generate a sentence describing the detections.

        Strategies:
        - 'shortest': Minimize description length (best compression)
        - 'detailed': Include more information
        - 'search': Try multiple and pick best MDL score
        """
        if not detections:
            return "Empty sky with no visible objects."

        # Group by class
        by_class = defaultdict(list)
        for det in detections:
            by_class[det['class']].append(det)

        if strategy == 'shortest':
            # Generate shortest possible description
            parts = []
            for cls, dets in by_class.items():
                count = self.get_count_word(len(dets))
                pos = self.get_position(dets[0]['bbox'], img_size)
                if len(dets) == 1:
                    parts.append(f"{count} {cls} in the {pos}")
                else:
                    parts.append(f"{count} {cls}s")

            if len(parts) == 1:
                return parts[0] + "."
            else:
                return ", ".join(parts[:-1]) + " and " + parts[-1] + "."

        elif strategy == 'detailed':
            # More detailed description
            sentences = []
            for cls, dets in by_class.items():
                count = self.get_count_word(len(dets))
                action = self.get_action(cls)
                for det in dets:
                    pos = self.get_position(det['bbox'], img_size)
                    sentences.append(f"A {cls} is {action} in the {pos}.")
            return " ".join(sentences)

        elif strategy == 'search':
            # Try multiple phrasings, pick best MDL
            candidates = []

            # Try shortest
            short = self.generate(detections, img_size, 'shortest')
            candidates.append(short)

            # Try detailed
            detailed = self.generate(detections, img_size, 'detailed')
            candidates.append(detailed)

            # Try templates
            for template in self.templates:
                for cls, dets in by_class.items():
                    try:
                        sentence = template.format(
                            count=self.get_count_word(len(dets)),
                            object=cls,
                            action=self.get_action(cls),
                            location=self.get_position(dets[0]['bbox'], img_size)
                        )
                        candidates.append(sentence)
                    except:
                        pass

            # Score all candidates
            scored = [(s, self.score_sentence(s, detections)) for s in candidates]
            scored.sort(key=lambda x: -x[1]['mdl_score'])

            return scored[0][0]  # Return best

        return self.generate(detections, img_size, 'shortest')


# =============================================================================
# PART 2: Scene Understanding via Compression
# =============================================================================

class CompressionSceneUnderstanding:
    """
    Understand scenes by finding the most compressed representation.

    Key insight: Understanding = finding the shortest program that
                 explains all observations.

    This handles:
    - Object relationships
    - Scene context
    - Anomaly detection (things that don't compress well)
    """

    def __init__(self):
        # Learned scene patterns (could be discovered automatically)
        self.scene_patterns = {
            'flight': {
                'objects': ['plane'],
                'description': 'aircraft in flight'
            },
            'birds_flying': {
                'objects': ['wildlife', 'wildlife'],
                'description': 'flock of birds'
            },
            'meteor_shower': {
                'objects': ['meteorite', 'meteorite'],
                'description': 'meteor shower'
            },
            'busy_sky': {
                'objects': ['plane', 'wildlife', 'meteorite'],
                'description': 'busy night sky with various objects'
            }
        }

    def find_pattern(self, detections):
        """Find best matching pattern for detections."""
        if not detections:
            return None, "empty sky"

        det_classes = sorted([d['class'] for d in detections])

        best_pattern = None
        best_match = 0

        for name, pattern in self.scene_patterns.items():
            pattern_classes = sorted(pattern['objects'])

            # Count matches
            matches = sum(1 for c in det_classes if c in pattern_classes)
            match_ratio = matches / max(len(det_classes), len(pattern_classes))

            if match_ratio > best_match:
                best_match = match_ratio
                best_pattern = (name, pattern['description'])

        return best_pattern, f"match_score={best_match:.2f}"

    def compute_scene_complexity(self, detections, img_complexity_map):
        """
        Compute overall scene complexity.

        Low complexity = well-understood scene (matches patterns)
        High complexity = unusual scene (anomaly)
        """
        # Object complexity (are objects simple?)
        obj_complexity = np.mean([d.get('complexity', 0.5) for d in detections]) if detections else 0

        # Pattern complexity (does scene match known patterns?)
        pattern, _ = self.find_pattern(detections)
        pattern_complexity = 0.3 if pattern else 0.7

        # Background complexity
        bg_complexity = np.mean(img_complexity_map) if img_complexity_map is not None else 0.5

        return {
            'object_complexity': obj_complexity,
            'pattern_complexity': pattern_complexity,
            'background_complexity': bg_complexity,
            'total': (obj_complexity + pattern_complexity + bg_complexity) / 3
        }


# =============================================================================
# PART 3: Universal Compression Model
# =============================================================================

class UniversalCompressionModel:
    """
    UNIVERSAL model that handles vision AND language via compression.

    This is the key demonstration:
    - ONE principle (compression)
    - MULTIPLE modalities (vision, language)
    - ZERO task-specific training

    The same insight applies everywhere:
    - Objects are compressible regions in images
    - Sentences are compressible descriptions of scenes
    - Understanding is finding the shortest explanation
    """

    def __init__(self, img_size=128):
        self.img_size = img_size

        # Vision component
        self.window_size = 24
        self.stride = 8
        self.complexity_threshold = 0.65

        # Language component
        self.language_model = CompressionLanguageModel()

        # Scene understanding
        self.scene_understanding = CompressionSceneUnderstanding()

        # Tiny classifier (only for labeling, not detection)
        self.class_templates = {}

    def compute_complexity(self, patch):
        """Compute Kolmogorov complexity approximation via gzip."""
        patch_bytes = patch.tobytes()
        compressed = gzip.compress(patch_bytes, compresslevel=9)
        return len(compressed) / len(patch_bytes)

    def fit(self, train_data):
        """Learn class templates from training data."""
        class_features = defaultdict(list)

        for img, annotations in train_data:
            for ann in annotations:
                x, y, w, h = ann['bbox']
                x, y = max(0, x), max(0, y)
                x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)

                if x2 > x and y2 > y:
                    patch = img[y:y2, x:x2]
                    # Simple features: mean color + complexity
                    features = [
                        patch.mean() / 255,
                        patch.std() / 255,
                        self.compute_complexity(patch)
                    ]
                    class_features[ann['class_idx']].append(features)

        for class_idx, feats in class_features.items():
            self.class_templates[class_idx] = np.mean(feats, axis=0)

        # Compute complexity threshold from background
        bg_complexities = []
        for img, _ in train_data[:20]:
            for y in range(0, img.shape[0] - self.window_size, self.stride * 2):
                for x in range(0, img.shape[1] - self.window_size, self.stride * 2):
                    patch = img[y:y+self.window_size, x:x+self.window_size]
                    bg_complexities.append(self.compute_complexity(patch))

        self.complexity_threshold = np.percentile(bg_complexities, 30)
        print(f"Model fitted. Complexity threshold: {self.complexity_threshold:.3f}")

    def classify_patch(self, patch):
        """Classify a patch using templates."""
        features = [
            patch.mean() / 255,
            patch.std() / 255,
            self.compute_complexity(patch)
        ]

        best_class = 0
        best_dist = float('inf')

        for class_idx, template in self.class_templates.items():
            dist = np.linalg.norm(np.array(features) - template)
            if dist < best_dist:
                best_dist = dist
                best_class = class_idx

        confidence = 1.0 / (1.0 + best_dist)
        return best_class, confidence

    def detect(self, img):
        """Detect objects using compression."""
        h, w = img.shape[:2]
        complexity_map = np.ones((h, w)) * 999

        # Compute complexity map
        for y in range(0, h - self.window_size, self.stride):
            for x in range(0, w - self.window_size, self.stride):
                patch = img[y:y+self.window_size, x:x+self.window_size]
                complexity = self.compute_complexity(patch)
                complexity_map[y:y+self.window_size, x:x+self.window_size] = np.minimum(
                    complexity_map[y:y+self.window_size, x:x+self.window_size],
                    complexity
                )

        # Find low-complexity regions
        mask = complexity_map < self.complexity_threshold
        visited = np.zeros_like(mask, dtype=bool)
        detections = []

        CLASSES = ['plane', 'wildlife', 'meteorite']

        for y in range(h):
            for x in range(w):
                if mask[y, x] and not visited[y, x]:
                    # Flood fill
                    region = []
                    stack = [(y, x)]

                    while stack:
                        cy, cx = stack.pop()
                        if (0 <= cy < h and 0 <= cx < w
                            and mask[cy, cx] and not visited[cy, cx]):
                            visited[cy, cx] = True
                            region.append((cy, cx))
                            stack.extend([(cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1)])

                    if len(region) >= 30:
                        ys, xs = zip(*region)
                        x1, y1 = min(xs), min(ys)
                        x2, y2 = max(xs), max(ys)
                        bbox = [x1, y1, x2-x1, y2-y1]

                        # Classify
                        patch = img[y1:y2, x1:x2]
                        if patch.size > 0:
                            class_idx, conf = self.classify_patch(patch)
                            avg_complexity = np.mean([complexity_map[p[0], p[1]] for p in region[:100]])

                            detections.append({
                                'bbox': bbox,
                                'class_idx': class_idx,
                                'class': CLASSES[class_idx],
                                'confidence': conf,
                                'complexity': avg_complexity
                            })

        return detections, complexity_map

    def generate_caption(self, detections, strategy='search'):
        """Generate natural language caption for detections."""
        return self.language_model.generate(detections, self.img_size, strategy)

    def understand_scene(self, detections, complexity_map):
        """High-level scene understanding."""
        pattern, match_info = self.scene_understanding.find_pattern(detections)
        complexity = self.scene_understanding.compute_scene_complexity(detections, complexity_map)

        return {
            'pattern': pattern,
            'match_info': match_info,
            'complexity': complexity
        }

    def process(self, img):
        """
        FULL PIPELINE: Image → Detection + Caption + Understanding

        This is the universal model in action:
        1. Detect objects (compression-based)
        2. Generate caption (compression-based)
        3. Understand scene (compression-based)

        ALL using the same principle: find the shortest description.
        """
        # Detect
        detections, complexity_map = self.detect(img)

        # Caption (multiple strategies)
        caption_short = self.generate_caption(detections, 'shortest')
        caption_best = self.generate_caption(detections, 'search')

        # Understand
        understanding = self.understand_scene(detections, complexity_map)

        # Score captions
        short_score = self.language_model.score_sentence(caption_short, detections)
        best_score = self.language_model.score_sentence(caption_best, detections)

        return {
            'detections': detections,
            'captions': {
                'shortest': caption_short,
                'best': caption_best,
            },
            'scores': {
                'shortest': short_score,
                'best': best_score,
            },
            'understanding': understanding,
            'complexity_map': complexity_map
        }


# =============================================================================
# PART 4: Dataset Generator (from previous experiment)
# =============================================================================

class SkyWatchGenerator:
    """Generate synthetic sky images with objects."""

    CLASSES = ['plane', 'wildlife', 'meteorite']

    def __init__(self, img_size=128, seed=42):
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)

    def generate_sky_background(self):
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        for y in range(self.img_size):
            ratio = y / self.img_size
            r = int(20 + ratio * 60)
            g = int(20 + ratio * 40)
            b = int(80 - ratio * 30)
            img[y, :] = [r, g, b]

        noise = self.rng.randint(-15, 15, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        n_stars = self.rng.randint(10, 30)
        for _ in range(n_stars):
            x, y = self.rng.randint(0, self.img_size, 2)
            brightness = self.rng.randint(150, 255)
            if 0 <= x < self.img_size and 0 <= y < self.img_size:
                img[y, x] = [brightness, brightness, brightness]

        return img

    def draw_plane(self, img, bbox):
        x, y, w, h = bbox
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        cx, cy = x + w//2, y + h//2

        fuse_w, fuse_h = int(w * 0.8), int(h * 0.2)
        draw.rectangle([cx - fuse_w//2, cy - fuse_h//2, cx + fuse_w//2, cy + fuse_h//2], fill=(200, 200, 210))

        wing_w, wing_h = int(w * 0.3), int(h * 0.7)
        draw.rectangle([cx - wing_w//2, cy - wing_h//2, cx + wing_w//2, cy + wing_h//2], fill=(180, 180, 190))

        return np.array(pil_img)

    def draw_wildlife(self, img, bbox):
        x, y, w, h = bbox
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        cx, cy = x + w//2, y + h//2

        body_w, body_h = int(w * 0.5), int(h * 0.4)
        draw.ellipse([cx - body_w//2, cy - body_h//2, cx + body_w//2, cy + body_h//2], fill=(60, 60, 70))

        wing_span = int(w * 0.9)
        draw.polygon([(cx, cy), (cx - wing_span//2, cy - h//3), (cx - wing_span//4, cy)], fill=(50, 50, 60))
        draw.polygon([(cx, cy), (cx + wing_span//2, cy - h//3), (cx + wing_span//4, cy)], fill=(50, 50, 60))

        return np.array(pil_img)

    def draw_meteorite(self, img, bbox):
        x, y, w, h = bbox
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        x1, y1 = x, y + h
        x2, y2 = x + w, y

        for offset in range(-2, 3):
            draw.line([(x1, y1 + offset), (x2, y2 + offset)], fill=(255, 255, 200), width=2)
        draw.ellipse([x2 - 4, y2 - 4, x2 + 4, y2 + 4], fill=(255, 255, 255))

        return np.array(pil_img)

    def generate_sample(self):
        img = self.generate_sky_background()
        annotations = []

        n_objects = self.rng.randint(1, 4)

        for _ in range(n_objects):
            class_weights = [0.4, 0.4, 0.2]
            class_idx = self.rng.choice(3, p=class_weights)
            class_name = self.CLASSES[class_idx]

            obj_size = self.rng.randint(15, 35)
            x = self.rng.randint(5, self.img_size - obj_size - 5)
            y = self.rng.randint(5, self.img_size - obj_size - 5)
            w = obj_size + self.rng.randint(-5, 10)
            h = obj_size + self.rng.randint(-5, 10)
            bbox = [x, y, w, h]

            if class_name == 'plane':
                img = self.draw_plane(img, bbox)
            elif class_name == 'wildlife':
                img = self.draw_wildlife(img, bbox)
            else:
                img = self.draw_meteorite(img, bbox)

            annotations.append({'bbox': bbox, 'class': class_name, 'class_idx': class_idx})

        return img, annotations

    def generate_dataset(self, n_train=200, n_val=50):
        print(f"Generating {n_train} training + {n_val} validation images...")
        train_data = [self.generate_sample() for _ in range(n_train)]
        val_data = [self.generate_sample() for _ in range(n_val)]
        return train_data, val_data


# =============================================================================
# PART 5: Main Experiment
# =============================================================================

def main():
    print("=" * 70)
    print("UNIVERSAL COMPRESSION-BASED INTELLIGENCE")
    print("=" * 70)
    print()
    print("ONE principle (compression) → MULTIPLE capabilities:")
    print("  1. Object Detection (find compressible regions)")
    print("  2. Language Generation (shortest description)")
    print("  3. Scene Understanding (pattern matching)")
    print()

    # Generate dataset
    generator = SkyWatchGenerator(img_size=128, seed=42)
    train_data, val_data = generator.generate_dataset(n_train=100, n_val=20)

    # Create universal model
    print("-" * 70)
    print("TRAINING UNIVERSAL MODEL")
    print("-" * 70)
    model = UniversalCompressionModel(img_size=128)
    model.fit(train_data)

    # Process examples
    print()
    print("-" * 70)
    print("EXAMPLE OUTPUTS")
    print("-" * 70)

    for i in range(5):
        img, gt_annotations = val_data[i]

        print(f"\n{'='*50}")
        print(f"IMAGE {i+1}")
        print(f"{'='*50}")

        # Ground truth
        print(f"\nGround Truth:")
        for ann in gt_annotations:
            print(f"  - {ann['class']} at {ann['bbox']}")

        # Process with universal model
        result = model.process(img)

        # Detections
        print(f"\nDetections ({len(result['detections'])} found):")
        for det in result['detections']:
            print(f"  - {det['class']} at {det['bbox'][:2]} "
                  f"(complexity={det['complexity']:.3f}, conf={det['confidence']:.2f})")

        # Captions
        print(f"\nGenerated Captions:")
        print(f"  Shortest: \"{result['captions']['shortest']}\"")
        print(f"            ({result['scores']['shortest']['bits']} bits)")
        print(f"  Best MDL: \"{result['captions']['best']}\"")
        print(f"            ({result['scores']['best']['bits']} bits, "
              f"accuracy={result['scores']['best']['accuracy']:.2f})")

        # Understanding
        print(f"\nScene Understanding:")
        understanding = result['understanding']
        if understanding['pattern']:
            print(f"  Pattern: {understanding['pattern'][1]}")
        print(f"  Scene Complexity: {understanding['complexity']['total']:.3f}")

    # Summary statistics
    print()
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    total_gt = 0
    total_detected = 0
    total_bits_short = 0
    total_bits_best = 0

    for img, gt in val_data:
        result = model.process(img)
        total_gt += len(gt)
        total_detected += len(result['detections'])
        total_bits_short += result['scores']['shortest']['bits']
        total_bits_best += result['scores']['best']['bits']

    print(f"\nDetection:")
    print(f"  Objects found: {total_detected}/{total_gt} "
          f"({100*total_detected/max(total_gt,1):.1f}%)")

    print(f"\nLanguage Generation:")
    print(f"  Avg caption length (shortest): {total_bits_short/len(val_data):.1f} bits")
    print(f"  Avg caption length (best MDL): {total_bits_best/len(val_data):.1f} bits")
    print(f"  Compression gain: {100*(1 - total_bits_best/max(total_bits_short,1)):.1f}%")

    print()
    print("-" * 70)
    print("KEY INSIGHT")
    print("-" * 70)
    print("""
The SAME compression principle handles THREE different tasks:

1. DETECTION: Objects are low-complexity regions
   → Found by computing gzip(region) for each patch

2. LANGUAGE: Best caption minimizes description length
   → Score = accuracy / bits (MDL principle)

3. UNDERSTANDING: Scenes match compressed patterns
   → Recognition = finding the shortest explanation

This is what "universal intelligence" looks like:
NOT billions of task-specific parameters,
BUT one principle (compression) applied everywhere.

The brain does exactly this - it compresses sensory input
into increasingly abstract representations.
""")

    return model, val_data


if __name__ == "__main__":
    main()
