#!/usr/bin/env python3
"""
Quick test script to verify the ViT MAE fixes work properly.
"""

import torch
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.init_vitmae_fixed import init_vit_mae_fixed
from src.utils.set_device import set_device

def test_vitmae_initialization():
    """Test if the ViT MAE model initializes correctly and can handle a forward pass."""
    
    device = torch.device(set_device())
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing ViT MAE model...")
    model = init_vit_mae_fixed()
    model.to(device)
    model.train()
    
    # Create test input: [batch_size, channels, height, width]
    batch_size = 2
    channels = 4  # 4 stacked frames
    height = 84
    width = 84
    
    test_input = torch.randn(batch_size, channels, height, width).to(device)
    print(f"Test input shape: {test_input.shape}")
    print(f"Test input range: [{test_input.min().item():.3f}, {test_input.max().item():.3f}]")
    
    # Normalize input to [0, 1] range like real data
    test_input = torch.clamp(test_input * 0.5 + 0.5, 0, 1)
    print(f"Normalized input range: [{test_input.min().item():.3f}, {test_input.max().item():.3f}]")
    
    # Forward pass
    print("Performing forward pass...")
    with torch.no_grad():
        outputs = model(test_input)
        
    print(f"✅ Forward pass successful!")
    print(f"   Loss: {outputs.loss.item():.4f}")
    print(f"   Logits shape: {outputs.logits.shape}")
    print(f"   Logits range: [{outputs.logits.min().item():.3f}, {outputs.logits.max().item():.3f}]")
    print(f"   Mask shape: {outputs.mask.shape}")
    print(f"   Mask ratio: {outputs.mask.float().mean().item():.3f} (should be ~0.75)")
    
    # Test backward pass for gradient check
    print("Testing backward pass...")
    model.zero_grad()
    outputs = model(test_input)
    loss = outputs.loss
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm().item()
            grad_norms.append(grad_norm)
    
    if grad_norms:
        mean_grad = sum(grad_norms) / len(grad_norms)
        max_grad = max(grad_norms)
        min_grad = min(grad_norms)
        print(f"✅ Gradient check successful!")
        print(f"   Mean gradient norm: {mean_grad:.2e}")
        print(f"   Max gradient norm: {max_grad:.2e}")
        print(f"   Min gradient norm: {min_grad:.2e}")
        print(f"   Total parameters with gradients: {len(grad_norms)}")
        
        if mean_grad > 1e-8:
            print("✅ Gradients look healthy!")
        else:
            print("⚠️  Warning: Gradients are very small")
    else:
        print("❌ No gradients found!")
    
    print("\n🎉 Test completed successfully!")

if __name__ == "__main__":
    test_vitmae_initialization()
