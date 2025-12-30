#!/usr/bin/env python3
"""
Download Skywatch4 YOLOv8 dataset from Roboflow.

Dataset contains 3 classes (Plane, WildLife, meteorite) with:
- 7088 training images
- 1436 validation images
- 342 test images

Source: https://app.roboflow.com/sky-orxqi/skywatch4/2
"""

import subprocess
import sys


def install_roboflow():
    """Install roboflow package if not available."""
    try:
        import roboflow
    except ImportError:
        print("Installing roboflow package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow", "-q"])


def download_dataset(output_dir: str = "."):
    """Download the Skywatch4 dataset."""
    install_roboflow()

    from roboflow import Roboflow

    rf = Roboflow(api_key="ZV0qo3Axk6PkGPzOnYLl")
    project = rf.workspace("sky-orxqi").project("skywatch4")
    version = project.version(2)
    dataset = version.download("yolov8", location=output_dir)

    print(f"\nDataset downloaded to: {dataset.location}")
    print("\nDataset structure:")
    print("  - train/images/ (7088 images)")
    print("  - train/labels/ (7088 labels)")
    print("  - valid/images/ (1436 images)")
    print("  - valid/labels/ (1436 labels)")
    print("  - test/images/ (342 images)")
    print("  - test/labels/ (342 labels)")
    print("  - data.yaml (dataset configuration)")

    return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Skywatch4 YOLOv8 dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./Skywatch4-2",
        help="Output directory for the dataset (default: ./Skywatch4-2)"
    )
    args = parser.parse_args()

    download_dataset(args.output_dir)
