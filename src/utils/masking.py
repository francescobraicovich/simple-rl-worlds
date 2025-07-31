# masking.py

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


def patchify(images: Tensor, patch_size: int) -> Tensor:
    """
    Convert images into flattened patch embeddings.

    Args:
        images: (B, C, H, W) tensor of images.
        patch_size: size of each square patch P.

    Returns:
        patches: (B, N, P*P*C) tensor, where N = (H/P)*(W/P).
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"Image dimensions ({H}x{W}) must be divisible by patch size ({patch_size})"
    p = patch_size
    gh, gw = H // p, W // p
    # reshape into (B, C, gh, p, gw, p)
    x = images.view(B, C, gh, p, gw, p)
    # reorder to (B, gh, gw, p, p, C)
    x = x.permute(0, 2, 4, 3, 5, 1)
    # flatten to (B, gh*gw, p*p*C)
    patches = x.reshape(B, gh * gw, p * p * C)
    return patches


def unpatchify(patches: Tensor, patch_size: int, image_size: int) -> Tensor:
    """
    Reconstruct images from flattened patch embeddings.

    Args:
        patches: (B, N, P*P*C) tensor.
        patch_size: size of each square patch P.
        image_size: original image height/width H (== W).

    Returns:
        images: (B, C, H, W) tensor.
    """
    B, N, patch_dim = patches.shape
    p = patch_size
    gh = gw = image_size // p
    assert gh * gw == N, f"Number of patches ({N}) doesn't match image size/{p}"
    C = patch_dim // (p * p)
    # reshape to (B, gh, gw, p, p, C)
    x = patches.reshape(B, gh, gw, p, p, C)
    # reorder to (B, C, gh, p, gw, p)
    x = x.permute(0, 5, 1, 3, 2, 4)
    # merge to (B, C, H, W)
    images = x.reshape(B, C, gh * p, gw * p)
    return images


def random_masking(
    x: Tensor,
    mask_ratio: float,
    return_uncertainties: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Perform per-sample random masking by shuffling patch tokens.

    Args:
        x: (B, N, D) sequence of patch embeddings.
        mask_ratio: fraction of patches to mask (0 <= mask_ratio <= 1).
        return_uncertainties: if True, also return the raw random noise.

    Returns:
        x_masked: (B, N_keep, D) embeddings of unmasked (kept) patches.
        mask: (B, N) binary mask (1 for masked, 0 for keep).
        ids_keep: (B, N_keep) indices of kept patches.
        ids_restore: (B, N) indices to restore original ordering.
        [optional] noise: (B, N) raw random values used for shuffling.
    """
    B, N, D = x.shape
    assert 0.0 <= mask_ratio <= 1.0, "mask_ratio must be in [0, 1]"
    len_keep = int(N * (1 - mask_ratio))

    # 1. generate random noise for each patch
    noise = torch.rand(B, N, device=x.device)

    # 2. sort noise for each sample to get shuffle order
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # 3. keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]

    # 4. mask out the rest
    x_masked = torch.gather(
        x, dim=1,
        index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
    )

    # 5. generate binary mask: 0 = keep, 1 = remove
    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to align with original x
    mask = torch.gather(mask, dim=1, index=ids_restore)

    if return_uncertainties:
        return x_masked, mask, ids_keep, ids_restore, noise
    else:
        return x_masked, mask, ids_keep, ids_restore


def get_masked_image(
    images: Tensor,
    mask: Tensor,
    patch_size: int
) -> Tensor:
    """
    Zero-out masked patches in the image for visualization.

    Args:
        images: (B, C, H, W) original images.
        mask: (B, N) binary mask (1 for masked).
        patch_size: size of patches P.

    Returns:
        masked_images: (B, C, H, W) with masked patches set to zero.
    """
    B, C, H, W = images.shape
    # patchify the images
    patches = patchify(images, patch_size)      # (B, N, P*P*C)
    # mask out
    mask = mask.unsqueeze(-1)                   # (B, N, 1)
    patches = patches * (1.0 - mask)             # zero masked
    # reconstruct
    masked_images = unpatchify(patches, patch_size, H)
    return masked_images


def compute_reconstruction_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor
) -> Tensor:
    """
    Compute MSE loss on masked patches only.

    Args:
        pred: (B, N, D) reconstructed patch embeddings or pixels.
        target: (B, N, D) ground-truth.
        mask: (B, N) binary mask (1 for masked).

    Returns:
        loss: scalar MSE over masked positions.
    """
    # expand mask to match pred shape
    mask_expanded = mask.unsqueeze(-1).expand_as(pred)
    diff = (pred - target) * mask_expanded
    # average only over masked elements
    loss = (diff ** 2).sum() / mask_expanded.sum().clamp(min=1.0)
    return loss


# Optional: utility to split mask into 2D for visualization
def mask_to_boolean_grid(
    mask: Tensor,
    patch_size: int,
    image_size: int
) -> Tensor:
    """
    Convert a 1D patch mask to a 2D boolean mask image.

    Args:
        mask: (B, N) binary mask per patch.
        patch_size: P.
        image_size: H == W.

    Returns:
        grid: (B, 1, H, W) boolean mask (1 for masked pixels).
    """
    B, N = mask.shape
    gh = gw = image_size // patch_size
    # reshape to (B, gh, gw)
    grid = mask.reshape(B, gh, gw)
    # expand each True to patch_size square
    grid = grid.unsqueeze(-1).unsqueeze(-1)  # (B, gh, gw, 1, 1)
    grid = grid.expand(-1, -1, -1, patch_size, patch_size)
    # to image
    grid = grid.reshape(B, 1, image_size, image_size).bool()
    return grid
