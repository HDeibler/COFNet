"""
Quick training script for COFNet on SkyWatch dataset.

Optimized for MacBook Pro (MPS or CPU).
Uses small subset and lightweight model for fast iteration.

Usage:
    python scripts/train_quick.py
"""

import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from models.cofnet import COFNet
from data.coco_dataset import COCODataset, collate_fn
from training.ssl import ScaleContrastiveLearning, CrossScaleReconstruction


def get_device():
    """Get best available device for Mac."""
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def create_mini_dataset(full_dataset, num_samples=30):
    """Create a small subset for quick testing."""
    indices = list(range(min(num_samples, len(full_dataset))))
    return Subset(full_dataset, indices)


def compute_losses(model, outputs, targets, device):
    """Compute detection losses."""
    pred_boxes = outputs['pred_boxes']
    pred_logits = outputs['pred_logits']

    B = pred_boxes.shape[0]
    total_cls_loss = torch.tensor(0.0, device=device)
    total_box_loss = torch.tensor(0.0, device=device)

    for b in range(B):
        gt_boxes = targets[b]['boxes'].to(device)
        gt_labels = targets[b]['labels'].to(device)

        if len(gt_boxes) == 0:
            continue

        # Simple matching: for each GT, find closest prediction
        # (In production, use Hungarian matching)
        pred_b = pred_boxes[b]  # [num_queries, 4]
        logits_b = pred_logits[b]  # [num_queries, num_classes]

        for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            # Find closest prediction by L1 distance
            dists = (pred_b - gt_box.unsqueeze(0)).abs().sum(dim=-1)
            closest_idx = dists.argmin()

            # Box loss (L1)
            box_loss = nn.functional.l1_loss(pred_b[closest_idx], gt_box)
            total_box_loss = total_box_loss + box_loss

            # Classification loss
            cls_loss = nn.functional.cross_entropy(
                logits_b[closest_idx].unsqueeze(0),
                gt_label.unsqueeze(0)
            )
            total_cls_loss = total_cls_loss + cls_loss

    return total_cls_loss, total_box_loss


def train_epoch(model, loader, optimizer, ssl_modules, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        images = batch['images'].to(device)
        targets = batch['targets']

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, targets)

        # Detection losses
        diffusion_loss = outputs.get('loss_diffusion', torch.tensor(0.0, device=device))
        cls_loss, box_loss = compute_losses(model, outputs, targets, device)

        # SSL losses (lightweight for speed)
        scl_loss = torch.tensor(0.0, device=device)
        csr_loss = torch.tensor(0.0, device=device)

        if ssl_modules is not None and batch_idx % 2 == 0:  # Every other batch
            scl, csr = ssl_modules
            scl_outputs = scl(model, images)
            csr_outputs = csr(model, images)
            scl_loss = scl_outputs['total'] * 0.1
            csr_loss = csr_outputs['total'] * 0.1

        # Total loss
        loss = diffusion_loss + cls_loss + 5.0 * box_loss + scl_loss + csr_loss

        # Backward
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)}: "
                  f"loss={loss.item():.4f} "
                  f"(diff={diffusion_loss.item():.3f}, "
                  f"cls={cls_loss.item():.3f}, "
                  f"box={box_loss.item():.3f})")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, device):
    """Quick validation."""
    model.eval()

    total_box_error = 0.0
    num_boxes = 0

    for batch in loader:
        images = batch['images'].to(device)
        targets = batch['targets']

        outputs = model(images)
        pred_boxes = outputs['pred_boxes']

        for b, target in enumerate(targets):
            gt_boxes = target['boxes'].to(device)
            if len(gt_boxes) == 0:
                continue

            # Average box error (simple metric)
            for gt_box in gt_boxes:
                dists = (pred_boxes[b] - gt_box.unsqueeze(0)).abs().sum(dim=-1)
                min_dist = dists.min()
                total_box_error += min_dist.item()
                num_boxes += 1

    avg_error = total_box_error / max(num_boxes, 1)
    return avg_error


def main():
    print("\n" + "="*60)
    print("COFNet Quick Training on SkyWatch Dataset")
    print("="*60)

    # Config
    DATA_ROOT = Path("/Users/deibler/Documents/projects/Train/data/processed")
    NUM_TRAIN_SAMPLES = 30  # Small subset
    NUM_VAL_SAMPLES = 10
    IMAGE_SIZE = (256, 256)  # Smaller for speed
    BATCH_SIZE = 2
    NUM_EPOCHS = 3
    LR = 1e-4
    USE_SSL = True

    device = get_device()

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = COCODataset(
        img_folder=str(DATA_ROOT / "train" / "images"),
        ann_file=str(DATA_ROOT / "train_coco.json"),
        image_size=IMAGE_SIZE,
    )

    val_dataset = COCODataset(
        img_folder=str(DATA_ROOT / "valid" / "images"),
        ann_file=str(DATA_ROOT / "valid_coco.json"),
        image_size=IMAGE_SIZE,
    )

    # Create mini subsets
    train_subset = create_mini_dataset(train_dataset, NUM_TRAIN_SAMPLES)
    val_subset = create_mini_dataset(val_dataset, NUM_VAL_SAMPLES)

    print(f"Training on {len(train_subset)} images")
    print(f"Validating on {len(val_subset)} images")

    # Data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues on Mac
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Create model (small config for speed)
    print("\nCreating model...")
    model = COFNet(
        num_classes=3,  # Plane, WildLife, meteorite
        backbone_dims=[32, 64, 128, 256],  # Small backbone
        csf_dim=64,
        num_queries=20,
        diffusion_steps_train=50,  # Fewer steps
        diffusion_steps_infer=4,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # SSL modules
    ssl_modules = None
    if USE_SSL:
        print("Enabling SSL modules...")
        scl = ScaleContrastiveLearning(feature_dim=64, num_scales=4).to(device)
        csr = CrossScaleReconstruction(feature_dim=64, num_scales=4).to(device)
        ssl_modules = (scl, csr)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Training loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print("-"*60)

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, ssl_modules, device, epoch
        )

        # Validate
        val_error = validate(model, val_loader, device)

        elapsed = time.time() - start_time

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Box Error: {val_error:.4f}")
        print(f"  Time: {elapsed:.1f}s")

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    # Save model
    save_path = Path(__file__).parent.parent / "output" / "quick_train.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()
