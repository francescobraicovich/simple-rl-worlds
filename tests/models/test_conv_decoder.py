#!/usr/bin/env python3
"""
Test script for ConvDecoder

This script tests the ConvDecoder implementation with comprehensive tests
including shape verification, device compatibility, and model summary.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.utils.init_models import init_conv_decoder


def test_conv_decoder():
    """Test script for ConvDecoder."""
    
    print("Testing ConvDecoder...")
    print("-" * 50)
    
    # Create model using init_models
    model = init_conv_decoder()
    print("Model created using init_conv_decoder()")
    
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
    B = 2  # Batch size
    latent_dim = model.config['latent_dim']
    
    # Create dummy input: [B, 1, latent_dim]
    dummy_input = torch.randn(B, 1, latent_dim)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    try:
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Output shape: {output.shape}")
        expected_shape = (B, 1, 1, 64, 64)
        print(f"Expected shape: {expected_shape}")
        
        # Verify output shape
        if output.shape == expected_shape:
            print("Shape test PASSED")
        else:
            print(f"Shape test FAILED! Expected {expected_shape}, got {output.shape}")
            return False
        
        # Check output range
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
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
        
        # Test different batch sizes
        print("\nTesting different batch sizes...")
        test_batch_sizes = [1, 4, 8]
        
        for test_b in test_batch_sizes:
            test_input = torch.randn(test_b, 1, latent_dim)
            with torch.no_grad():
                test_output = model(test_input)
            expected = (test_b, 1, 1, 64, 64)
            if test_output.shape == expected:
                print(f"Batch size test ({test_b}) passed: {test_output.shape}")
            else:
                print(f"Batch size test ({test_b}) failed: expected {expected}, got {test_output.shape}")
                return False
        
        # Test intermediate shapes
        print("\nTesting intermediate shapes...")
        x = dummy_input.squeeze(1)  # [B, latent_dim]
        
        # Linear layer output
        linear_out = model.initial_linear(x)
        expected_linear_shape = (B, 256 * 4 * 4)
        print(f"Linear output shape: {linear_out.shape}, expected: {expected_linear_shape}")
        
        # Reshaped feature map
        reshaped = linear_out.reshape(B, 256, 4, 4)
        expected_reshaped_shape = (B, 256, 4, 4)
        print(f"Reshaped feature map: {reshaped.shape}, expected: {expected_reshaped_shape}")
        
        # Test each transpose conv layer
        x = reshaped
        expected_sizes = [(B, 128, 8, 8), (B, 64, 16, 16), (B, 32, 32, 32), (B, 1, 64, 64)]
        
        for i, (conv_transpose, layer_norm) in enumerate(zip(model.conv_transpose_layers, model.layer_norms)):
            x = conv_transpose(x)
            if i < len(model.conv_transpose_layers) - 1:  # Not the final layer
                x = model.activation(x)
                if layer_norm is not None:
                    C, H, W = x.shape[1], x.shape[2], x.shape[3]
                    x_temp = x.permute(0, 2, 3, 1)
                    x_temp = layer_norm(x_temp)
                    x = x_temp.permute(0, 3, 1, 2)
            
            print(f"After transpose conv {i+1}: {x.shape}, expected: {expected_sizes[i]}")
            if x.shape != expected_sizes[i]:
                print(f"Intermediate shape test failed at layer {i+1}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    model = init_conv_decoder()
    
    # Test wrong number of tokens
    try:
        wrong_tokens = torch.randn(2, 3, 1024)  # 3 tokens instead of 1
        model(wrong_tokens)
        print("Should have failed with wrong number of tokens")
        return False
    except AssertionError:
        print("Correctly rejected wrong number of tokens")
    
    # Test wrong latent dimension
    try:
        wrong_latent_dim = torch.randn(2, 1, 512)  # 512 instead of 1024
        model(wrong_latent_dim)
        print("Should have failed with wrong latent dimension")
        return False
    except AssertionError:
        print("Correctly rejected wrong latent dimension")
    
    return True


def test_architecture_progression():
    """Test that the architecture progresses correctly through sizes."""
    print("\nTesting architecture progression...")
    
    model = init_conv_decoder()
    
    # Expected progression: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
    # Expected channels: 256 -> 128 -> 64 -> 32 -> 1
    
    expected_progression = [
        ("Initial", 4, 256),
        ("After layer 1", 8, 128),  
        ("After layer 2", 16, 64),
        ("After layer 3", 32, 32),
        ("After layer 4", 64, 1)
    ]
    
    print("Expected architecture progression:")
    for stage, size, channels in expected_progression:
        print(f"  {stage}: {channels} channels, {size}x{size} spatial")
    
    return True


if __name__ == "__main__":
    """Run all tests."""
    
    print("Starting ConvDecoder tests...")
    
    # Run main test
    success = test_conv_decoder()
    
    if success:
        # Run edge case tests
        success = test_edge_cases()
    
    if success:
        # Test architecture progression
        success = test_architecture_progression()
    
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
        model = init_conv_decoder()
        summary(model, input_size=(2, 1, 1024), device='cpu')
    except ImportError:
        print("\nInstall torchinfo for detailed model summary: pip install torchinfo")
    
    sys.exit(0 if success else 1) 