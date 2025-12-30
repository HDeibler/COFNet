"""
COCO Format Dataset for COFNet.

Supports standard COCO-format annotations and custom datasets.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import get_train_transforms, get_val_transforms


class COCODataset(Dataset):
    """
    COCO Format Dataset.

    Args:
        img_folder: Path to image folder
        ann_file: Path to annotation JSON file (COCO format)
        image_size: Target image size as (height, width)
        transforms: Optional albumentation transforms
        is_train: Whether this is training (affects default transforms)
    """

    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        image_size: Tuple[int, int] = (1280, 1280),
        transforms: Optional[Callable] = None,
        is_train: bool = True,
    ):
        self.img_folder = Path(img_folder)
        self.ann_file = Path(ann_file)
        self.image_size = image_size
        self.is_train = is_train

        # Load annotations
        self.images, self.annotations, self.categories = self._load_annotations()

        # Build image id to annotations mapping
        self.img_to_anns = self._build_img_to_anns()

        # Build category id mapping (COCO ids may not be contiguous)
        self.cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_cat_id = {idx: cat['id'] for idx, cat in enumerate(self.categories)}

        # Set transforms
        if transforms is not None:
            self.transforms = transforms
        else:
            if is_train:
                self.transforms = get_train_transforms(image_size)
            else:
                self.transforms = get_val_transforms(image_size)

    def _load_annotations(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load COCO format annotations."""
        with open(self.ann_file, 'r') as f:
            data = json.load(f)

        images = data.get('images', [])
        annotations = data.get('annotations', [])
        categories = data.get('categories', [])

        return images, annotations, categories

    def _build_img_to_anns(self) -> Dict[int, List[Dict]]:
        """Build mapping from image id to annotations."""
        img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        return img_to_anns

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            dict with:
                - images: [3, H, W] tensor
                - targets: dict with 'boxes' [N, 4] and 'labels' [N]
        """
        img_info = self.images[idx]
        img_id = img_info['id']

        # Load image
        img_path = self.img_folder / img_info['file_name']
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])

        # Parse boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            # Skip crowd annotations
            if ann.get('iscrowd', 0):
                continue

            # COCO format: [x, y, width, height]
            x, y, w, h = ann['bbox']

            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue

            # Convert to [x1, y1, x2, y2] for albumentations
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_idx[ann['category_id']])

        # Convert to numpy
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        # Apply transforms
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels,
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)

            # Handle empty boxes after transform
            if len(boxes) == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)

        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            # Image should be [H, W, C] numpy -> [C, H, W] tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Convert boxes to (cx, cy, w, h) normalized format
        boxes_tensor = torch.from_numpy(boxes).float()
        if len(boxes_tensor) > 0:
            boxes_tensor = self._xyxy_to_cxcywh(boxes_tensor, self.image_size)

        labels_tensor = torch.from_numpy(labels).long()

        return {
            'images': image,
            'targets': {
                'boxes': boxes_tensor,
                'labels': labels_tensor,
                'image_id': img_id,
            }
        }

    def _xyxy_to_cxcywh(self, boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """
        Convert boxes from [x1, y1, x2, y2] to normalized [cx, cy, w, h].

        Args:
            boxes: [N, 4] boxes in [x1, y1, x2, y2] format
            image_size: (height, width) of image

        Returns:
            [N, 4] boxes in normalized [cx, cy, w, h] format
        """
        h, w = image_size

        x1, y1, x2, y2 = boxes.unbind(-1)

        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        return torch.stack([cx, cy, bw, bh], dim=-1).clamp(0, 1)

    @property
    def num_classes(self) -> int:
        """Number of classes in dataset."""
        return len(self.categories)

    def get_category_names(self) -> List[str]:
        """Get list of category names."""
        return [cat['name'] for cat in self.categories]


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for variable-sized targets.

    Args:
        batch: List of samples from dataset

    Returns:
        dict with batched images and list of targets
    """
    images = torch.stack([item['images'] for item in batch], dim=0)
    targets = [item['targets'] for item in batch]

    return {
        'images': images,
        'targets': targets,
    }


class SkyWatchDataset(COCODataset):
    """
    SkyWatch Dataset - specialized for aerial object detection.

    Inherits from COCODataset with SkyWatch-specific defaults.
    Classes: plane, wildlife, meteorite
    """

    CLASS_NAMES = ['plane', 'wildlife', 'meteorite']

    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        image_size: Tuple[int, int] = (1280, 1280),
        transforms: Optional[Callable] = None,
        is_train: bool = True,
    ):
        super().__init__(
            img_folder=img_folder,
            ann_file=ann_file,
            image_size=image_size,
            transforms=transforms,
            is_train=is_train,
        )

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse frequency class weights for handling imbalance.

        Returns:
            Tensor of class weights
        """
        class_counts = torch.zeros(self.num_classes)

        for anns in self.img_to_anns.values():
            for ann in anns:
                if not ann.get('iscrowd', 0):
                    cat_idx = self.cat_id_to_idx[ann['category_id']]
                    class_counts[cat_idx] += 1

        # Inverse frequency weighting
        total = class_counts.sum()
        weights = total / (class_counts + 1e-6)

        # Normalize
        weights = weights / weights.sum() * self.num_classes

        return weights
