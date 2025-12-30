#!/usr/bin/env python3
"""
COFNet Training Script

Usage:
    python scripts/train.py --config configs/cofnet_skywatch.yaml
    python scripts/train.py --config configs/cofnet_base.yaml --pretrain path/to/weights.pth
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.cofnet import COFNet, build_cofnet
from data.coco_dataset import COCODataset, collate_fn
from training.trainer import COFNetTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train COFNet")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--pretrain", "-p",
        type=str,
        default=None,
        help="Path to pretrained weights for fine-tuning",
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, or cuda:N)",
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load and merge config files."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle base config inheritance
    if '__base__' in config:
        base_path = Path(config_path).parent / config['__base__']
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        # Merge: config overrides base
        base_config = deep_merge(base_config, config)
        del base_config['__base__']
        config = base_config

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def build_dataloaders(config: dict) -> tuple:
    """Build train and validation dataloaders."""
    data_config = config.get('data', {})

    # Image size
    image_size = tuple(data_config.get('image_size', [640, 640]))

    # Train dataset
    train_dataset = COCODataset(
        img_folder=data_config['train_img_folder'],
        ann_file=data_config['train_ann_file'],
        image_size=image_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get('training', {}).get('batch_size', 8),
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Val dataset
    val_loader = None
    if 'val_img_folder' in data_config and 'val_ann_file' in data_config:
        val_dataset = COCODataset(
            img_folder=data_config['val_img_folder'],
            ann_file=data_config['val_ann_file'],
            image_size=image_size,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.get('training', {}).get('batch_size', 8),
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=True,
        )

    return train_loader, val_loader


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    opt_config = config.get('optimizer', {})
    train_config = config.get('training', {})

    opt_type = opt_config.get('type', 'AdamW')
    lr = train_config.get('lr', 1e-4)
    weight_decay = train_config.get('weight_decay', 1e-4)

    if opt_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(opt_config.get('betas', [0.9, 0.999])),
        )
    elif opt_type == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=opt_config.get('momentum', 0.9),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build learning rate scheduler from config."""
    sched_config = config.get('scheduler', {})
    train_config = config.get('training', {})

    sched_type = sched_config.get('type', 'CosineAnnealingLR')
    epochs = train_config.get('epochs', 100)

    if sched_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config.get('T_max', epochs),
            eta_min=sched_config.get('eta_min', 1e-6),
        )
    elif sched_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config.get('step_size', 30),
            gamma=sched_config.get('gamma', 0.1),
        )
    elif sched_type == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sched_config.get('milestones', [60, 80]),
            gamma=sched_config.get('gamma', 0.1),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")

    return scheduler


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Load config
    config = load_config(args.config)

    print("=" * 60)
    print("COFNet Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Pretrain: {args.pretrain}")
    print(f"Resume: {args.resume}")
    print(f"Seed: {args.seed}")

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    print("=" * 60)

    # Build model
    model_config = config.get('model', {})
    model = build_cofnet(model_config)
    print(f"\nModel: COFNet")
    print(f"  Classes: {model_config.get('num_classes', 80)}")
    print(f"  Backbone dims: {model_config.get('backbone_dims', [96, 192, 384, 768])}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load pretrained weights
    if args.pretrain:
        print(f"\nLoading pretrained weights from {args.pretrain}")
        state_dict = torch.load(args.pretrain, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        # Load with strict=False to allow mismatched keys
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    # Build dataloaders
    print("\nBuilding dataloaders...")
    train_loader, val_loader = build_dataloaders(config)
    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Val batches: {len(val_loader)}")

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    print(f"\nOptimizer: {type(optimizer).__name__}")
    print(f"Scheduler: {type(scheduler).__name__}")

    # Build trainer config
    train_config = config.get('training', {})
    loss_config = config.get('loss', {})
    trainer_config = {
        'epochs': train_config.get('epochs', 100),
        'use_amp': train_config.get('use_amp', True),
        'gradient_clip': train_config.get('gradient_clip', 1.0),
        'log_interval': config.get('logging', {}).get('log_interval', 50),
        'save_interval': config.get('logging', {}).get('save_interval', 5),
        'diffusion_weight': loss_config.get('diffusion_weight', 1.0),
        'classification_weight': loss_config.get('classification_weight', 1.0),
        'box_l1_weight': loss_config.get('box_l1_weight', 5.0),
        'giou_weight': loss_config.get('giou_weight', 2.0),
        'output_dir': config.get('output_dir', './output'),
    }

    # Build trainer
    trainer = COFNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=trainer_config,
        device=device,
    )

    # Resume from checkpoint
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Initialize wandb
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=config.get('logging', {}).get('wandb_project', 'cofnet'),
                entity=config.get('logging', {}).get('wandb_entity'),
                config=config,
            )
            print("\nWandB initialized")
        except ImportError:
            print("\nWandB not installed, skipping...")

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()


if __name__ == "__main__":
    main()
