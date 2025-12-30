"""
Data Augmentation Pipeline for COFNet.

Uses albumentations for efficient augmentations with bounding box support.
"""

from typing import Callable, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    image_size: Tuple[int, int] = (1280, 1280),
    mosaic: bool = True,
    color_jitter: float = 0.4,
    random_flip: float = 0.5,
) -> Callable:
    """
    Get training augmentation pipeline.

    Args:
        image_size: Target (height, width)
        mosaic: Enable mosaic augmentation (handled separately)
        color_jitter: Color jitter strength
        random_flip: Horizontal flip probability

    Returns:
        Albumentations Compose transform
    """
    h, w = image_size

    transforms = [
        # Resize to target size
        A.LongestMaxSize(max_size=max(h, w)),
        A.PadIfNeeded(
            min_height=h,
            min_width=w,
            border_mode=0,
            value=(114, 114, 114),  # Gray padding
        ),

        # Geometric augmentations
        A.HorizontalFlip(p=random_flip),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=0,
            value=(114, 114, 114),
            p=0.5,
        ),

        # Color augmentations
        A.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
            hue=color_jitter * 0.1,
            p=0.5,
        ),
        A.ToGray(p=0.1),  # Occasional grayscale
        A.GaussianBlur(blur_limit=(3, 7), p=0.1),
        A.GaussNoise(var_limit=(10, 50), p=0.1),

        # Normalize
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),

        # Convert to tensor
        ToTensorV2(),
    ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',  # [x1, y1, x2, y2]
            label_fields=['labels'],
            min_area=1,
            min_visibility=0.1,
        ),
    )


def get_val_transforms(
    image_size: Tuple[int, int] = (1280, 1280),
) -> Callable:
    """
    Get validation/test augmentation pipeline (minimal transforms).

    Args:
        image_size: Target (height, width)

    Returns:
        Albumentations Compose transform
    """
    h, w = image_size

    transforms = [
        # Resize to target size
        A.LongestMaxSize(max_size=max(h, w)),
        A.PadIfNeeded(
            min_height=h,
            min_width=w,
            border_mode=0,
            value=(114, 114, 114),
        ),

        # Normalize
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),

        # Convert to tensor
        ToTensorV2(),
    ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=1,
            min_visibility=0.1,
        ),
    )


def get_mosaic_transform(
    image_size: Tuple[int, int] = (1280, 1280),
) -> Callable:
    """
    Get mosaic augmentation (4-image composition).

    Note: Mosaic is typically applied in the dataloader, not per-sample.
    This returns a post-mosaic transform for cleaning up the composed image.

    Args:
        image_size: Target (height, width)

    Returns:
        Albumentations Compose transform for post-mosaic processing
    """
    h, w = image_size

    transforms = [
        A.RandomCrop(height=h, width=w, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=1,
            min_visibility=0.1,
        ),
    )


def get_strong_augmentation(
    image_size: Tuple[int, int] = (1280, 1280),
) -> Callable:
    """
    Get strong augmentation pipeline for semi-supervised learning.

    Used in SDSS pretraining for creating harder augmented views.

    Args:
        image_size: Target (height, width)

    Returns:
        Albumentations Compose transform
    """
    h, w = image_size

    transforms = [
        # Resize
        A.LongestMaxSize(max_size=max(h, w)),
        A.PadIfNeeded(
            min_height=h,
            min_width=w,
            border_mode=0,
            value=(114, 114, 114),
        ),

        # Strong geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.3,
            rotate_limit=30,
            border_mode=0,
            value=(114, 114, 114),
            p=0.7,
        ),
        A.Perspective(scale=(0.05, 0.15), p=0.3),

        # Strong color
        A.ColorJitter(
            brightness=0.6,
            contrast=0.6,
            saturation=0.6,
            hue=0.1,
            p=0.8,
        ),
        A.ToGray(p=0.2),
        A.Posterize(num_bits=4, p=0.2),
        A.Equalize(p=0.2),
        A.Solarize(threshold=128, p=0.2),

        # Noise and blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 11)),
            A.MotionBlur(blur_limit=(3, 11)),
            A.MedianBlur(blur_limit=7),
        ], p=0.3),
        A.GaussNoise(var_limit=(10, 100), p=0.3),

        # Dropout
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            fill_value=114,
            p=0.3,
        ),

        # Normalize
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=1,
            min_visibility=0.1,
        ),
    )


def get_scale_augmentation(
    image_size: Tuple[int, int] = (1280, 1280),
    scale_range: Tuple[float, float] = (0.5, 2.0),
) -> Callable:
    """
    Get scale-specific augmentation for scale-contrastive learning.

    Creates views at different scales for learning scale-equivariant features.

    Args:
        image_size: Target (height, width)
        scale_range: Range of scales to sample from

    Returns:
        Albumentations Compose transform
    """
    h, w = image_size

    transforms = [
        # Random scale
        A.RandomScale(
            scale_limit=(scale_range[0] - 1, scale_range[1] - 1),
            p=1.0,
        ),

        # Crop or pad to target size
        A.LongestMaxSize(max_size=max(h, w)),
        A.PadIfNeeded(
            min_height=h,
            min_width=w,
            border_mode=0,
            value=(114, 114, 114),
        ),

        # Light augmentation
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
            p=0.5,
        ),

        # Normalize
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=1,
            min_visibility=0.1,
        ),
    )
