#!/usr/bin/env python3
"""
Test script for ConvEncoder

This script tests the ConvEncoder implementation with comprehensive tests
including shape verification, device compatibility, and model summary.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.utils.init_models import init_conv_encoder


def test_conv_encoder():
    """Test script for ConvEncoder."""
    
    print("Testing ConvEncoder...")
    print("-" * 50)
    
    # Create model using init_models
    model = init_conv_encoder()
    print("Model created using init_conv_encoder()")
    
    # Print model summary
    print("\nModel Architecture:")
    print(model)
    
    # Print configuration
    print(f"\nModel Configuration:")
    for key, value in model.config.items():
        print(f"  {key}: {value}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Test with dummy input
    print("\nTesting forward pass...")
    
    # Test parameters
    B, T = 4, 8  # Batch size 4, sequence length 8
    
    # Create dummy input: (B, 1, T, 64, 64)
    dummy_input = torch.randn(B, 1, T, 64, 64)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    try:
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: ({B}, {T}, {model.config['latent_dim']})")
        
        # Verify output shape
        expected_shape = (B, T, model.config['latent_dim'])
        if output.shape == expected_shape:
            print("Shape test PASSED")
        else:
            print(f"Shape test FAILED! Expected {expected_shape}, got {output.shape}")
            return False
        
        # Test device compatibility
        if torch.cuda.is_available():
            print("\nTesting CUDA compatibility...")
            model_cuda = model.cuda()
            dummy_input_cuda = dummy_input.cuda()
            
            with torch.no_grad():
                output_cuda = model_cuda(dummy_input_cuda)
            
            print(f"CUDA test passed! Output shape: {output_cuda.shape}")
            # Move back to CPU for next test
            model = model.cpu()
        
        if torch.backends.mps.is_available():
            print("\nTesting MPS compatibility...")
            model_mps = model.to('mps')
            dummy_input_mps = dummy_input.to('mps')
            
            with torch.no_grad():
                output_mps = model_mps(dummy_input_mps)
            
            print(f"MPS test passed! Output shape: {output_mps.shape}")
            # Move back to CPU
            model = model.cpu()
        
        # Test different batch sizes and sequence lengths
        print("\nTesting different input sizes...")
        test_cases = [
            (1, 1),   # Single batch, single frame
            (2, 4),   # Small batch, short sequence
            (8, 16),  # Larger batch, longer sequence
        ]
        
        for test_b, test_t in test_cases:
            test_input = torch.randn(test_b, 1, test_t, 64, 64)
            with torch.no_grad():
                test_output = model(test_input)
            expected = (test_b, test_t, model.config['latent_dim'])
            if test_output.shape == expected:
                print(f"Size test ({test_b}, {test_t}) passed: {test_output.shape}")
            else:
                print(f"Size test ({test_b}, {test_t}) failed: expected {expected}, got {test_output.shape}")
                return False
        
        # Test single frame encoding
        print("\nTesting single frame encoding...")
        single_frame = torch.randn(64, 64)
        with torch.no_grad():
            single_output = model.encode_single_frame(single_frame)
        
        expected_single_shape = (model.config['latent_dim'],)
        if single_output.shape == expected_single_shape:
            print(f"Single frame test passed: {single_output.shape}")
        else:
            print(f"Single frame test failed: expected {expected_single_shape}, got {single_output.shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    model = init_conv_encoder()
    
    # Test wrong input channels
    try:
        wrong_channels = torch.randn(2, 3, 4, 64, 64)  # 3 channels instead of 1
        model(wrong_channels)
        print("Should have failed with wrong number of channels")
        return False
    except AssertionError:
        print("Correctly rejected wrong number of channels")
    
    # Test wrong image size
    try:
        wrong_size = torch.randn(2, 1, 4, 32, 32)  # 32x32 instead of 64x64
        model(wrong_size)
        print("Should have failed with wrong image size")
        return False
    except AssertionError:
        print("Correctly rejected wrong image size")
    
    return True


if __name__ == "__main__":
    """Run all tests."""
    
    print("Starting ConvEncoder tests...")
    
    # Run main test
    success = test_conv_encoder()
    
    if success:
        # Run edge case tests
        success = test_edge_cases()
    
    print("\n" + "-" * 50)
    if success:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    
    print("Testing complete!")
    
    # Optional: Test with torchinfo if available
    try:
        from torchinfo import summary
        print("\nDetailed Model Summary:")
        model = init_conv_encoder()
        summary(model, input_size=(4, 1, 8, 64, 64), device='cpu')
    except ImportError:
        print("\nInstall torchinfo for detailed model summary: pip install torchinfo")
    
    sys.exit(0 if success else 1) 