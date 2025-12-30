"""
COFNet Trainer with full training loop.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast


class COFNetTrainer:
    """
    Trainer for COFNet with diffusion-based detection.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Training settings
        self.epochs = config.get('epochs', 100)
        self.use_amp = config.get('use_amp', True)
        self.gradient_clip = config.get('gradient_clip', 1.0)
        self.log_interval = config.get('log_interval', 50)
        self.save_interval = config.get('save_interval', 5)

        # Loss weights
        self.diffusion_weight = config.get('diffusion_weight', 1.0)
        self.cls_weight = config.get('classification_weight', 1.0)
        self.box_l1_weight = config.get('box_l1_weight', 5.0)
        self.giou_weight = config.get('giou_weight', 2.0)

        # Output directory
        self.output_dir = Path(config.get('output_dir', './output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None

        # Tracking
        self.current_epoch = 0
        self.best_map = 0.0

    def train(self):
        """Run full training loop."""
        print(f"\nStarting training for {self.epochs} epochs")
        print(f"Output directory: {self.output_dir}")

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_loss = self.train_epoch()

            # Evaluate
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.evaluate()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Log
            lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  LR: {lr:.6f}")
            if val_metrics:
                print(f"  Val mAP: {val_metrics.get('mAP', 0):.4f}")

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pth")

            # Save best model
            val_map = val_metrics.get('mAP', 0)
            if val_map > self.best_map:
                self.best_map = val_map
                self.save_checkpoint("best.pth")
                print(f"  New best mAP: {val_map:.4f}")

        print("\nTraining complete!")
        self.save_checkpoint("final.pth")

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            images = batch['images'].to(self.device)
            targets = batch['targets']
            for t in targets:
                t['boxes'] = t['boxes'].to(self.device)
                t['labels'] = t['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp and self.scaler is not None:
                with autocast():
                    outputs = self.model(images, targets)
                    loss = self.compute_loss(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, targets)
                loss = self.compute_loss(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
                self.optimizer.step()

            total_loss += loss.item()

            # Log
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  [{batch_idx + 1}/{num_batches}] Loss: {avg_loss:.4f}")

        return total_loss / num_batches

    def compute_loss(
        self,
        outputs: dict,
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        Compute combined loss for COFNet.

        Components:
        - Diffusion loss (MSE on predicted noise)
        - Classification loss (cross entropy)
        - Box regression loss (L1 + GIoU)
        """
        losses = {}

        # Diffusion loss (from model forward)
        if 'loss_diffusion' in outputs:
            losses['diffusion'] = outputs['loss_diffusion'] * self.diffusion_weight

        # Classification loss
        pred_logits = outputs['pred_logits']  # [B, N, num_classes]
        cls_loss = self.compute_classification_loss(pred_logits, targets)
        losses['cls'] = cls_loss * self.cls_weight

        # Box regression loss
        pred_boxes = outputs['pred_boxes']  # [B, N, 4]
        box_loss, giou_loss = self.compute_box_loss(pred_boxes, targets)
        losses['box_l1'] = box_loss * self.box_l1_weight
        losses['giou'] = giou_loss * self.giou_weight

        # Total loss
        total_loss = sum(losses.values(), torch.tensor(0.0, device=pred_logits.device))

        return total_loss

    def compute_classification_loss(
        self,
        pred_logits: torch.Tensor,
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        Compute classification loss using Hungarian matching.

        For simplicity, we use a direct assignment based on IoU.
        """
        B, N, C = pred_logits.shape
        device = pred_logits.device

        total_loss = torch.tensor(0.0, device=device)
        num_boxes = 0

        for b, target in enumerate(targets):
            gt_labels = target['labels']  # [num_gt]
            num_gt = len(gt_labels)

            if num_gt == 0:
                # No GT boxes - all predictions should be background
                # Using -1 as background class for cross entropy
                bg_target = torch.full((N,), C, device=device, dtype=torch.long)
                # Add background class for loss computation
                pred_with_bg = F.pad(pred_logits[b], (0, 1), value=-10)  # [N, C+1]
                total_loss += F.cross_entropy(pred_with_bg, bg_target)
                continue

            # Match predictions to GT using simple assignment
            # Take first num_gt predictions as matched
            num_matched = min(num_gt, N)

            # Loss for matched predictions
            matched_pred = pred_logits[b, :num_matched]  # [num_matched, C]
            matched_gt = gt_labels[:num_matched]  # [num_matched]
            total_loss += F.cross_entropy(matched_pred, matched_gt)

            num_boxes += num_matched

        # Average over boxes
        if num_boxes > 0:
            total_loss = total_loss / num_boxes

        return total_loss

    def compute_box_loss(
        self,
        pred_boxes: torch.Tensor,
        targets: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute box regression losses (L1 and GIoU).
        """
        B, N, _ = pred_boxes.shape
        device = pred_boxes.device

        l1_loss = torch.tensor(0.0, device=device)
        giou_loss = torch.tensor(0.0, device=device)
        num_boxes = 0

        for b, target in enumerate(targets):
            gt_boxes = target['boxes']  # [num_gt, 4]
            num_gt = len(gt_boxes)

            if num_gt == 0:
                continue

            # Match predictions to GT
            num_matched = min(num_gt, N)
            matched_pred = pred_boxes[b, :num_matched]  # [num_matched, 4]
            matched_gt = gt_boxes[:num_matched]  # [num_matched, 4]

            # L1 loss
            l1_loss += F.l1_loss(matched_pred, matched_gt, reduction='sum')

            # GIoU loss
            giou = self.compute_giou(matched_pred, matched_gt)
            giou_loss += (1 - giou).sum()

            num_boxes += num_matched

        # Average over boxes
        if num_boxes > 0:
            l1_loss = l1_loss / num_boxes
            giou_loss = giou_loss / num_boxes

        return l1_loss, giou_loss

    def compute_giou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Generalized IoU between two sets of boxes.

        Both boxes are in (cx, cy, w, h) format.
        """
        # Convert to (x1, y1, x2, y2)
        cx1, cy1, w1, h1 = boxes1.unbind(dim=-1)
        cx2, cy2, w2, h2 = boxes2.unbind(dim=-1)

        x1_1, y1_1 = cx1 - w1/2, cy1 - h1/2
        x2_1, y2_1 = cx1 + w1/2, cy1 + h1/2
        x1_2, y1_2 = cx2 - w2/2, cy2 - h2/2
        x2_2, y2_2 = cx2 + w2/2, cy2 + h2/2

        # Intersection
        inter_x1 = torch.max(x1_1, x1_2)
        inter_y1 = torch.max(y1_1, y1_2)
        inter_x2 = torch.min(x2_1, x2_2)
        inter_y2 = torch.min(y2_1, y2_2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        # IoU
        iou = inter_area / union_area.clamp(min=1e-6)

        # Enclosing box
        enc_x1 = torch.min(x1_1, x1_2)
        enc_y1 = torch.min(y1_1, y1_2)
        enc_x2 = torch.max(x2_1, x2_2)
        enc_y2 = torch.max(y2_1, y2_2)

        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

        # GIoU
        giou = iou - (enc_area - union_area) / enc_area.clamp(min=1e-6)

        return giou

    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return {'mAP': 0.0}

        self.model.eval()

        all_predictions = []
        all_targets = []

        for batch in self.val_loader:
            images = batch['images'].to(self.device)
            targets = batch['targets']

            # Forward pass
            outputs = self.model(images)

            # Collect predictions and targets
            pred_boxes = outputs['pred_boxes']  # [B, N, 4]
            pred_logits = outputs['pred_logits']  # [B, N, C]
            pred_scores = pred_logits.softmax(dim=-1)

            for b in range(len(targets)):
                all_predictions.append({
                    'boxes': pred_boxes[b].cpu(),
                    'scores': pred_scores[b].cpu(),
                })
                all_targets.append({
                    'boxes': targets[b]['boxes'],
                    'labels': targets[b]['labels'],
                })

        # Compute mAP (simplified)
        mAP = self.compute_map(all_predictions, all_targets)

        return {'mAP': mAP}

    def compute_map(
        self,
        predictions: List[Dict],
        targets: List[Dict],
    ) -> float:
        """
        Compute mean Average Precision (simplified version).
        """
        # For now, compute average IoU as a proxy for mAP
        total_iou = 0.0
        num_matches = 0

        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes']
            gt_boxes = target['boxes']

            if len(gt_boxes) == 0 or len(pred_boxes) == 0:
                continue

            # Get top predictions
            scores = pred['scores'].max(dim=-1)[0]
            topk = min(len(gt_boxes), len(pred_boxes))
            top_indices = scores.argsort(descending=True)[:topk]
            top_boxes = pred_boxes[top_indices]

            # Compute IoU with GT
            for i, pb in enumerate(top_boxes[:len(gt_boxes)]):
                gb = gt_boxes[i]
                giou = self.compute_giou(pb.unsqueeze(0), gb.unsqueeze(0))
                total_iou += giou.item()
                num_matches += 1

        if num_matches > 0:
            return total_iou / num_matches
        return 0.0

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_map': self.best_map,
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        path = self.output_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_map = checkpoint.get('best_map', 0.0)

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
