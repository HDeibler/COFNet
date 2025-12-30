"""
Quick test script to verify COFNet training pipeline works.

Run with: python tests/test_training.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn

def test_backbone():
    """Test MambaBackbone."""
    print("\n" + "="*60)
    print("Testing MambaBackbone...")
    print("="*60)

    from models.backbone.mamba_backbone import MambaBackbone

    # Small config for testing
    backbone = MambaBackbone(
        in_channels=3,
        dims=[32, 64, 128, 256],  # Smaller dims for testing
        depths=[1, 1, 2, 1],      # Fewer blocks
        d_state=8,
    )

    # Test forward
    x = torch.randn(2, 3, 128, 128)  # Small images
    features = backbone(x)

    print(f"  Input: {x.shape}")
    print(f"  Output features:")
    for i, f in enumerate(features):
        print(f"    Stage {i}: {f.shape}")

    # Verify shapes
    assert len(features) == 4
    assert features[0].shape == (2, 32, 32, 32)   # H/4
    assert features[1].shape == (2, 64, 16, 16)   # H/8
    assert features[2].shape == (2, 128, 8, 8)    # H/16
    assert features[3].shape == (2, 256, 4, 4)    # H/32

    print("  ✓ MambaBackbone OK")
    return backbone


def test_csf(backbone):
    """Test ContinuousScaleField."""
    print("\n" + "="*60)
    print("Testing ContinuousScaleField...")
    print("="*60)

    from models.csf.continuous_scale_field import ContinuousScaleField

    csf = ContinuousScaleField(
        backbone_dims=[32, 64, 128, 256],
        out_dim=64,
        scale_encoder='fourier',
    )

    # Get backbone features
    x = torch.randn(2, 3, 128, 128)
    backbone_features = backbone(x)

    # Test forward (unified features)
    unified = csf(backbone_features)
    print(f"  Unified features: {unified.shape}")
    assert unified.shape[0] == 2
    assert unified.shape[1] == 64  # out_dim

    # Test query_at_positions
    positions = torch.rand(2, 16, 2)  # 16 query positions
    scales = torch.rand(2, 16)        # 16 scale values
    queried = csf.query_at_positions(backbone_features, positions, scales)
    print(f"  Queried features: {queried.shape}")
    assert queried.shape == (2, 16, 64)

    # Test sample_at_boxes
    boxes = torch.rand(2, 10, 4)  # 10 boxes per image
    box_features = csf.sample_at_boxes(boxes, backbone_features)
    print(f"  Box features: {box_features.shape}")
    assert box_features.shape == (2, 10, 64)

    print("  ✓ ContinuousScaleField OK")
    return csf


def test_diffusion():
    """Test DiffusionBoxRefiner."""
    print("\n" + "="*60)
    print("Testing DiffusionBoxRefiner...")
    print("="*60)

    from models.diffusion.box_refiner import DiffusionBoxRefiner

    refiner = DiffusionBoxRefiner(
        feature_dim=64,
        num_steps=100,  # Fewer steps for testing
        num_heads=4,
        num_layers=2,
    )

    # Test forward (training mode)
    refiner.train()
    boxes = torch.rand(2, 10, 4)
    features = torch.randn(2, 64, 8, 8)  # Spatial features
    targets = [{'boxes': torch.rand(5, 4)} for _ in range(2)]

    refined, loss = refiner(boxes, features, targets)
    print(f"  Training - Refined boxes: {refined.shape}, Loss: {loss.item():.4f}")
    assert refined.shape == (2, 10, 4)
    assert loss is not None

    # Test sample (inference mode)
    refiner.eval()
    with torch.no_grad():
        init_boxes = torch.rand(2, 10, 4)
        sampled = refiner.sample(init_boxes, features, num_steps=4)
    print(f"  Inference - Sampled boxes: {sampled.shape}")
    assert sampled.shape == (2, 10, 4)

    # Verify boxes are in valid range
    assert (sampled >= 0).all() and (sampled <= 1).all()

    print("  ✓ DiffusionBoxRefiner OK")
    return refiner


def test_cofnet():
    """Test full COFNet model."""
    print("\n" + "="*60)
    print("Testing COFNet (full model)...")
    print("="*60)

    from models.cofnet import COFNet

    model = COFNet(
        num_classes=3,
        backbone_dims=[32, 64, 128, 256],
        csf_dim=64,
        num_queries=20,
        diffusion_steps_train=100,
        diffusion_steps_infer=4,
    )

    # Test training forward
    model.train()
    images = torch.randn(2, 3, 128, 128)
    targets = [
        {'boxes': torch.rand(3, 4), 'labels': torch.randint(0, 3, (3,))},
        {'boxes': torch.rand(5, 4), 'labels': torch.randint(0, 3, (5,))},
    ]

    outputs = model(images, targets)
    print(f"  Training outputs:")
    print(f"    pred_boxes: {outputs['pred_boxes'].shape}")
    print(f"    pred_logits: {outputs['pred_logits'].shape}")
    print(f"    loss_diffusion: {outputs.get('loss_diffusion', 'N/A')}")

    # Test inference forward
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    print(f"  Inference outputs:")
    print(f"    pred_boxes: {outputs['pred_boxes'].shape}")
    print(f"    pred_logits: {outputs['pred_logits'].shape}")

    print("  ✓ COFNet OK")
    return model


def test_ssl_modules(model):
    """Test SSL modules."""
    print("\n" + "="*60)
    print("Testing SSL Modules...")
    print("="*60)

    from training.ssl import (
        ScaleContrastiveLearning,
        CrossScaleReconstruction,
        DiffusionConvergenceDiscovery,
    )

    images = torch.randn(2, 3, 128, 128)

    # Test ScaleContrastiveLearning
    print("\n  Testing ScaleContrastiveLearning...")
    scl = ScaleContrastiveLearning(
        feature_dim=64,
        num_scales=8,
    )
    scl_outputs = scl(model, images)
    print(f"    Losses: {list(scl_outputs.keys())}")
    print(f"    Total loss: {scl_outputs['total'].item():.4f}")
    assert 'total' in scl_outputs
    print("    ✓ ScaleContrastiveLearning OK")

    # Test CrossScaleReconstruction
    print("\n  Testing CrossScaleReconstruction...")
    csr = CrossScaleReconstruction(
        feature_dim=64,
        num_scales=8,
        mask_ratio=0.5,
    )
    csr_outputs = csr(model, images)
    print(f"    Losses: {list(csr_outputs.keys())}")
    print(f"    Total loss: {csr_outputs['total'].item():.4f}")
    assert 'total' in csr_outputs
    print("    ✓ CrossScaleReconstruction OK")

    # Test DiffusionConvergenceDiscovery
    print("\n  Testing DiffusionConvergenceDiscovery...")
    dcd = DiffusionConvergenceDiscovery(
        num_initializations=2,  # Fewer for speed
        min_cluster_size=1,
    )
    model.eval()
    with torch.no_grad():
        dcd_outputs = dcd(model, images, num_queries=10)
    print(f"    Outputs: {list(dcd_outputs.keys())}")
    print(f"    Discovered boxes: {dcd_outputs['discovered_boxes'].shape}")
    print(f"    Objectness scores: {dcd_outputs['objectness_scores'].shape}")
    print("    ✓ DiffusionConvergenceDiscovery OK")

    print("\n  ✓ All SSL modules OK")


def test_training_step(model):
    """Test a single training step."""
    print("\n" + "="*60)
    print("Testing Training Step...")
    print("="*60)

    from training.ssl import ScaleContrastiveLearning, CrossScaleReconstruction

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # SSL modules
    scl = ScaleContrastiveLearning(feature_dim=64, num_scales=8)
    csr = CrossScaleReconstruction(feature_dim=64, num_scales=8)

    # Dummy batch
    images = torch.randn(2, 3, 128, 128)
    targets = [
        {'boxes': torch.rand(3, 4), 'labels': torch.randint(0, 3, (3,))},
        {'boxes': torch.rand(5, 4), 'labels': torch.randint(0, 3, (5,))},
    ]

    print("  Running 3 training iterations...")
    for step in range(3):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, targets)

        # Compute losses
        diffusion_loss = outputs.get('loss_diffusion', torch.tensor(0.0))
        scl_loss = scl(model, images)['total']
        csr_loss = csr(model, images)['total']

        # Classification loss (simplified)
        pred_logits = outputs['pred_logits']
        cls_loss = torch.tensor(0.0)

        # Total loss
        total_loss = diffusion_loss + 0.1 * scl_loss + 0.1 * csr_loss

        # Backward
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update
        optimizer.step()

        print(f"    Step {step+1}: loss={total_loss.item():.4f} "
              f"(diff={diffusion_loss.item():.4f}, scl={scl_loss.item():.4f}, csr={csr_loss.item():.4f})")

    print("  ✓ Training step OK")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("COFNet Training Pipeline Test")
    print("="*60)

    # Set seed for reproducibility
    torch.manual_seed(42)

    try:
        # Test individual components
        backbone = test_backbone()
        csf = test_csf(backbone)
        refiner = test_diffusion()

        # Test full model
        model = test_cofnet()

        # Test SSL modules
        test_ssl_modules(model)

        # Test training step
        test_training_step(model)

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
