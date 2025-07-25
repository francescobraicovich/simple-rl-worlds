"""
Utility functions for handling video patches and masking operations.

This module provides functions to extract and reassemble video tubelets
for use in Masked Autoencoder (MAE) pretraining and other video processing tasks.
"""

import torch


def extract_tubelets(x: torch.Tensor) -> torch.Tensor:
    """
    Extract tubelets from input video tensor by unfolding temporal and spatial dimensions.
    
    Args:
        x: Input tensor of shape [B, T, H, W] where:
           - B: batch size
           - T: temporal sequence length
           - H: height
           - W: width
    
    Returns:
        Tensor of patches shaped [B, N_tubelets, PATCH_T, PATCH_H, PATCH_W] where:
        - N_tubelets: total number of tubelets per sample
        - PATCH_T: temporal patch size (default: 2)
        - PATCH_H: spatial patch height (default: 4)
        - PATCH_W: spatial patch width (default: 4)
    """
    # Default patch sizes - these should match config.yaml values
    PATCH_T = 2  # patch_size_t from config
    PATCH_H = 4  # patch_size_h from config
    PATCH_W = 4  # patch_size_w from config
    
    B, T, H, W = x.shape
    
    # Ensure dimensions are divisible by patch sizes
    assert T % PATCH_T == 0, f"Temporal dimension {T} must be divisible by patch_size_t {PATCH_T}"
    assert H % PATCH_H == 0, f"Height {H} must be divisible by patch_size_h {PATCH_H}"
    assert W % PATCH_W == 0, f"Width {W} must be divisible by patch_size_w {PATCH_W}"
    
    # Calculate number of patches in each dimension
    n_patches_t = T // PATCH_T
    n_patches_h = H // PATCH_H
    n_patches_w = W // PATCH_W
    
    # Total number of tubelets per sample
    n_tubelets = n_patches_t * n_patches_h * n_patches_w
    
    # Reshape to extract tubelets
    # First, reshape to separate patch dimensions
    x = x.view(B, n_patches_t, PATCH_T, n_patches_h, PATCH_H, n_patches_w, PATCH_W)
    
    # Rearrange to group tubelets together: [B, n_patches_t, n_patches_h, n_patches_w, PATCH_T, PATCH_H, PATCH_W]
    x = x.permute(0, 1, 3, 5, 2, 4, 6)
    
    # Flatten the patch indices: [B, N_tubelets, PATCH_T, PATCH_H, PATCH_W]
    x = x.contiguous().view(B, n_tubelets, PATCH_T, PATCH_H, PATCH_W)
    
    return x


def reassemble_tubelets(patches: torch.Tensor) -> torch.Tensor:
    """
    Reassemble tubelets back into original tensor shape by folding patches.
    
    Args:
        patches: Tensor of patches shaped [B, N_tubelets, PATCH_T, PATCH_H, PATCH_W]
    
    Returns:
        Tensor of shape [B, T, H, W] where:
        - T = N_patches_t * PATCH_T
        - H = N_patches_h * PATCH_H  
        - W = N_patches_w * PATCH_W
    """
    # Default patch sizes - these should match config.yaml values
    PATCH_T = 2  # patch_size_t from config
    PATCH_H = 4  # patch_size_h from config
    PATCH_W = 4  # patch_size_w from config
    
    B, N_tubelets, patch_t, patch_h, patch_w = patches.shape
    
    # Verify patch dimensions match expected values
    assert patch_t == PATCH_T, f"Expected patch_t={PATCH_T}, got {patch_t}"
    assert patch_h == PATCH_H, f"Expected patch_h={PATCH_H}, got {patch_h}"
    assert patch_w == PATCH_W, f"Expected patch_w={PATCH_W}, got {patch_w}"
    
    # Calculate number of patches in each dimension
    # Assuming standard video dimensions: T=4, H=84, W=84
    T = 4  # sequence_length from config
    H = 84  # image_height from config
    W = 84  # image_width from config
    
    n_patches_t = T // PATCH_T
    n_patches_h = H // PATCH_H
    n_patches_w = W // PATCH_W
    
    # Verify total number of tubelets matches
    expected_tubelets = n_patches_t * n_patches_h * n_patches_w
    assert N_tubelets == expected_tubelets, f"Expected {expected_tubelets} tubelets, got {N_tubelets}"
    
    # Reshape to separate patch indices: [B, n_patches_t, n_patches_h, n_patches_w, PATCH_T, PATCH_H, PATCH_W]
    patches = patches.view(B, n_patches_t, n_patches_h, n_patches_w, PATCH_T, PATCH_H, PATCH_W)
    
    # Rearrange to original order: [B, n_patches_t, PATCH_T, n_patches_h, PATCH_H, n_patches_w, PATCH_W]
    patches = patches.permute(0, 1, 4, 2, 5, 3, 6)
    
    # Fold back to original shape: [B, T, H, W]
    x = patches.contiguous().view(B, T, H, W)
    
    return x


def generate_random_mask(batch_size: int, n_tubelets: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    """
    Generate random boolean masks for tubelet masking.
    
    Args:
        batch_size: Number of samples in the batch
        n_tubelets: Number of tubelets per sample
        mask_ratio: Fraction of tubelets to mask (between 0 and 1)
        device: Device to create the mask on
    
    Returns:
        Boolean tensor of shape [batch_size, n_tubelets] where True indicates masked tubelets
    """
    num_masked = int(n_tubelets * mask_ratio)
    
    # Create masks for each sample in the batch
    masks = []
    for _ in range(batch_size):
        # Create a mask with the specified number of True values
        mask = torch.zeros(n_tubelets, dtype=torch.bool, device=device)
        # Randomly select indices to mask
        masked_indices = torch.randperm(n_tubelets, device=device)[:num_masked]
        mask[masked_indices] = True
        masks.append(mask)
    
    return torch.stack(masks)


if __name__ == "__main__":
    # Test the functions
    print("Testing extract_tubelets and reassemble_tubelets...")
    
    # Create test tensor [B=2, T=4, H=84, W=84]
    x = torch.randn(2, 4, 84, 84)
    print(f"Original shape: {x.shape}")
    
    # Extract tubelets
    tubelets = extract_tubelets(x)
    print(f"Tubelets shape: {tubelets.shape}")
    
    # Reassemble
    x_reconstructed = reassemble_tubelets(tubelets)
    print(f"Reconstructed shape: {x_reconstructed.shape}")
    
    # Verify perfect reconstruction
    if torch.allclose(x, x_reconstructed):
        print("✓ Perfect reconstruction!")
    else:
        print("✗ Reconstruction error!")
        print(f"Max difference: {torch.max(torch.abs(x - x_reconstructed)).item()}")
    
    # Test masking
    print("\nTesting generate_random_mask...")
    batch_size, n_tubelets = 2, tubelets.shape[1]
    mask = generate_random_mask(batch_size, n_tubelets, mask_ratio=0.5, device='cpu')
    print(f"Mask shape: {mask.shape}")
    print(f"Mask ratio: {mask.float().mean().item():.3f}")
