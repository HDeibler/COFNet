"""
Scale-Diffusion Self-Supervision (SDSS) Pretrainer.

Orchestrates all SSL components for COFNet pretraining:
1. Diffusion Convergence Discovery (DCD) - Object discovery via diffusion
2. Scale-Contrastive Learning (SCL) - Scale-equivariant representations
3. Temporal-Diffusion Consistency (TDC) - Video consistency (optional)
4. Cross-Scale Reconstruction (CSR) - Masked scale modeling

This is a unified framework that exploits COFNet's unique architecture
(continuous scales + diffusion) for self-supervision.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from .diffusion_convergence import DiffusionConvergenceDiscovery, DiffusionConvergenceLoss
from .scale_contrastive import ScaleContrastiveLearning
from .temporal_diffusion import TemporalDiffusionConsistency
from .cross_scale_reconstruction import CrossScaleReconstruction

# Type alias for COFNet model (avoid circular imports)
COFNetModel = Any


class SDSSPretrainer:
    """
    Scale-Diffusion Self-Supervision Pretrainer for COFNet.

    Phases:
    1. Warm-up: Scale-contrastive + Cross-scale reconstruction (images only)
    2. Discovery: Add diffusion convergence discovery
    3. Temporal: Add temporal consistency (if video data available)

    This staged approach ensures stable training.
    """

    def __init__(
        self,
        model: COFNetModel,
        image_loader: DataLoader,
        video_loader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        device: torch.device = torch.device('cuda'),
    ):
        self.model: COFNetModel = model.to(device)
        self.image_loader = image_loader
        self.video_loader = video_loader
        self.device = device

        # Default config
        self.config = {
            'epochs': 100,
            'warmup_epochs': 10,
            'discovery_start_epoch': 20,
            'temporal_start_epoch': 40,
            'lr': 1e-4,
            'weight_decay': 0.05,
            'use_amp': True,
            'log_interval': 50,
            'save_interval': 10,
            'output_dir': './output/ssl_pretrain',
            # Loss weights
            'scale_contrastive_weight': 1.0,
            'cross_scale_weight': 1.0,
            'diffusion_convergence_weight': 0.5,
            'temporal_weight': 0.5,
        }
        if config:
            self.config.update(config)

        # Initialize SSL modules
        self.scale_contrastive = ScaleContrastiveLearning(
            feature_dim=model.csf.out_dim if hasattr(model, 'csf') else 256,
        ).to(device)

        self.cross_scale = CrossScaleReconstruction(
            feature_dim=model.csf.out_dim if hasattr(model, 'csf') else 256,
        ).to(device)

        self.diffusion_discovery = DiffusionConvergenceDiscovery().to(device)
        self.diffusion_loss = DiffusionConvergenceLoss().to(device)

        self.temporal_consistency = TemporalDiffusionConsistency().to(device)

        # Optimizer
        self.optimizer = optim.AdamW(
            list(model.parameters()) +
            list(self.scale_contrastive.parameters()) +
            list(self.cross_scale.parameters()),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'],
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs'],
            eta_min=1e-6,
        )

        # AMP
        self.scaler = GradScaler() if self.config['use_amp'] else None

        # Output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.current_epoch = 0

    def train(self):
        """Run full pretraining loop."""
        print(f"\n{'='*60}")
        print("Starting SDSS (Scale-Diffusion Self-Supervision) Pretraining")
        print(f"{'='*60}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Warmup epochs: {self.config['warmup_epochs']}")
        print(f"Discovery starts: epoch {self.config['discovery_start_epoch']}")
        print(f"Temporal starts: epoch {self.config['temporal_start_epoch']}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")

        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch

            # Determine which losses to use
            use_discovery = epoch >= self.config['discovery_start_epoch']
            use_temporal = (
                epoch >= self.config['temporal_start_epoch']
                and self.video_loader is not None
            )

            # Train one epoch
            losses = self.train_epoch(
                use_discovery=use_discovery,
                use_temporal=use_temporal,
            )

            # Update scheduler
            self.scheduler.step()

            # Log
            lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            print(f"  LR: {lr:.6f}")
            for name, value in losses.items():
                print(f"  {name}: {value:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pth")

        print("\nPretraining complete!")
        self.save_checkpoint("final.pth")

    def train_epoch(
        self,
        use_discovery: bool = False,
        use_temporal: bool = False,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.scale_contrastive.train()
        self.cross_scale.train()

        total_losses = {
            'scale_contrastive': 0.0,
            'cross_scale': 0.0,
            'total': 0.0,
        }
        if use_discovery:
            total_losses['discovery'] = 0.0
        if use_temporal:
            total_losses['temporal'] = 0.0

        num_batches = len(self.image_loader)

        for batch_idx, batch in enumerate(self.image_loader):
            # Get images
            if isinstance(batch, dict):
                images = batch['images'].to(self.device)
            else:
                images = batch[0].to(self.device)

            self.optimizer.zero_grad()

            if self.config['use_amp'] and self.scaler is not None:
                with autocast():
                    loss, batch_losses = self._compute_loss(
                        images,
                        use_discovery=use_discovery,
                        use_temporal=use_temporal,
                    )
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, batch_losses = self._compute_loss(
                    images,
                    use_discovery=use_discovery,
                    use_temporal=use_temporal,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            # Accumulate losses
            for name, value in batch_losses.items():
                if name in total_losses:
                    total_losses[name] += value

            # Log
            if (batch_idx + 1) % self.config['log_interval'] == 0:
                avg_loss = total_losses['total'] / (batch_idx + 1)
                print(f"  [{batch_idx + 1}/{num_batches}] Loss: {avg_loss:.4f}")

        # Average losses
        for name in total_losses:
            total_losses[name] /= num_batches

        return total_losses

    def _compute_loss(
        self,
        images: torch.Tensor,
        use_discovery: bool = False,
        use_temporal: bool = False,
    ) -> tuple:
        """Compute combined SSL loss."""
        batch_losses = {}

        # 1. Scale-Contrastive Loss
        scl_outputs = self.scale_contrastive(self.model, images)
        scl_loss = scl_outputs['total'] * self.config['scale_contrastive_weight']
        batch_losses['scale_contrastive'] = scl_loss.item()

        # 2. Cross-Scale Reconstruction Loss
        csr_outputs = self.cross_scale(self.model, images)
        csr_loss = csr_outputs['total'] * self.config['cross_scale_weight']
        batch_losses['cross_scale'] = csr_loss.item()

        total_loss = scl_loss + csr_loss

        # 3. Diffusion Convergence Discovery (after warmup)
        if use_discovery:
            with torch.no_grad():
                discoveries = self.diffusion_discovery(self.model, images)

            # Get model predictions
            with autocast(enabled=self.config['use_amp']):
                backbone_features = self.model.backbone(images)
                csf_features = self.model.csf(backbone_features)

                init_boxes = torch.rand(
                    images.shape[0], 100, 4, device=images.device
                )
                pred_boxes = self.model.box_refiner.sample(
                    init_boxes, csf_features, num_steps=4
                )
                pred_logits = self.model.cls_head(
                    self.model.csf.sample_at_boxes(pred_boxes, backbone_features)
                )

            predictions = {
                'pred_boxes': pred_boxes,
                'pred_logits': pred_logits,
            }

            disc_loss = self.diffusion_loss(predictions, discoveries)
            disc_loss = disc_loss * self.config['diffusion_convergence_weight']
            batch_losses['discovery'] = disc_loss.item()
            total_loss = total_loss + disc_loss

        # 4. Temporal Consistency (if video available)
        if use_temporal and self.video_loader is not None:
            # Get a video batch (sample from video loader)
            # This is a simplified approach - in practice you'd interleave
            try:
                video_batch = next(self._video_iter)
            except (StopIteration, AttributeError):
                self._video_iter = iter(self.video_loader)
                video_batch = next(self._video_iter)

            frame_t = video_batch['frame_t'].to(self.device)
            frame_t1 = video_batch['frame_t1'].to(self.device)
            flow = video_batch.get('flow')
            if flow is not None:
                flow = flow.to(self.device)

            with autocast(enabled=self.config['use_amp']):
                temp_outputs = self.temporal_consistency(
                    self.model, frame_t, frame_t1, flow
                )

            temp_loss = temp_outputs['total'] * self.config['temporal_weight']
            batch_losses['temporal'] = temp_loss.item()
            total_loss = total_loss + temp_loss

        batch_losses['total'] = total_loss.item()

        return total_loss, batch_losses

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scale_contrastive_state_dict': self.scale_contrastive.state_dict(),
            'cross_scale_state_dict': self.cross_scale.state_dict(),
            'config': self.config,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        path = self.output_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: Union[str, Path]):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1

        if 'scale_contrastive_state_dict' in checkpoint:
            self.scale_contrastive.load_state_dict(
                checkpoint['scale_contrastive_state_dict']
            )
        if 'cross_scale_state_dict' in checkpoint:
            self.cross_scale.load_state_dict(
                checkpoint['cross_scale_state_dict']
            )
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


class SDSSLoss(nn.Module):
    """
    Combined SDSS loss for end-to-end training.

    Can be used during supervised fine-tuning to maintain
    SSL objectives as regularization.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        scale_contrastive_weight: float = 0.1,
        cross_scale_weight: float = 0.1,
    ):
        super().__init__()
        self.scale_contrastive = ScaleContrastiveLearning(feature_dim=feature_dim)
        self.cross_scale = CrossScaleReconstruction(feature_dim=feature_dim)

        self.scale_contrastive_weight = scale_contrastive_weight
        self.cross_scale_weight = cross_scale_weight

    def forward(
        self,
        model: COFNetModel,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined SSL regularization loss."""
        scl_loss = self.scale_contrastive(model, images)['total']
        csr_loss = self.cross_scale(model, images)['total']

        return (
            self.scale_contrastive_weight * scl_loss +
            self.cross_scale_weight * csr_loss
        )
