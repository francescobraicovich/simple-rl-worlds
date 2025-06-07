import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOLoss(nn.Module):
    """
    A DINO-inspired auxiliary loss focusing on embedding centering.
    This component of DINO helps prevent model collapse by encouraging
    the batch-wise mean of embeddings to stay close to an EMA-updated center.

    Args:
        out_dim (int): The dimensionality of the embeddings this loss will operate on.
                       Required to initialize the center buffer.
        center_ema_decay (float): The decay rate for the EMA of the center.
                                  Defaults to 0.9.
        eps (float): Small epsilon for numerical stability if needed (not directly used in current formulation
                     but good practice for similar modules). Defaults to 1e-5.
    """
    def __init__(self, out_dim: int, center_ema_decay: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.out_dim = out_dim
        self.center_ema_decay = center_ema_decay
        self.eps = eps

        # Register buffer for the EMA center. This is not a model parameter.
        # It will be updated during training.
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def _update_center(self, batch_mean: torch.Tensor):
        """
        Update the EMA center with the mean of the current batch.
        Args:
            batch_mean (torch.Tensor): The mean of the current batch of embeddings.
        """
        self.center = self.center * self.center_ema_decay + batch_mean * (1 - self.center_ema_decay)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates the DINO-inspired centering loss.

        Args:
            z (torch.Tensor): Batch of embeddings, shape (N, D), where D must match self.out_dim.

        Returns:
            torch.Tensor: The centering loss value.
        """
        if z.ndim != 2 or z.shape[1] != self.out_dim:
            raise ValueError(
                f"Input tensor z must be 2D (batch_size, feature_dim) "
                f"and feature_dim must match out_dim={self.out_dim}. Got shape {z.shape}"
            )

        batch_mean = torch.mean(z, dim=0, keepdim=True) # Shape (1, D)

        # Calculate the loss: (batch_mean - self.center)^2
        # This encourages the batch mean to be close to the EMA center.
        centering_loss = (batch_mean - self.center.detach()).pow(2).sum() # Detach center as it's target

        # Update the center using EMA after computing the loss with the old center
        if self.training: # Only update center during training
             self._update_center(batch_mean.detach()) # Detach batch_mean for update if z requires grad

        return centering_loss

    def calculate_reg_terms(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method to align with VICReg's and BarlowTwins' usage in JEPA.
        For DINO centering, the primary loss is the centering loss itself.
        We return it as the total loss and the first component, with zero for the second.

        Args:
            z (torch.Tensor): Batch of embeddings, shape (N, D).

        Returns:
            torch.Tensor: The DINO centering loss value.
            torch.Tensor: The DINO centering loss value (as the first component).
            torch.Tensor: A zero tensor (as the second component, for API consistency).
        """
        loss = self.forward(z)
        # For API consistency with VICRegLoss/BarlowTwins (total, term1, term2)
        return loss, loss, torch.zeros_like(loss)


if __name__ == '__main__':
    # Example Usage
    batch_size, embed_dim = 128, 256
    dino_loss_fn = DINOLoss(out_dim=embed_dim, center_ema_decay=0.9)
    dino_loss_fn.train() # Set to training mode to update center

    print(f"Initial center: {dino_loss_fn.center}")

    # Simulate a few batches
    for i in range(5):
        z_embeddings = torch.randn(batch_size, embed_dim) + i # Shift mean over batches

        # Using calculate_reg_terms
        total_loss, main_loss_component, _ = dino_loss_fn.calculate_reg_terms(z_embeddings)
        # Alternatively, use forward directly:
        # total_loss = dino_loss_fn(z_embeddings)

        print(f"\nBatch {i+1}:")
        print(f"  Input z mean: {z_embeddings.mean(dim=0, keepdim=True)[0, :3]}...") # Print first 3 dims
        print(f"  Centering Loss: {total_loss.item():.4f}")
        print(f"  Updated center: {dino_loss_fn.center[0, :3]}...")

    # Test in eval mode (center should not update)
    dino_loss_fn.eval()
    print(f"\nCenter before eval batch (should remain same): {dino_loss_fn.center[0, :3]}...")
    z_eval_embeddings = torch.randn(batch_size, embed_dim) + 10
    eval_loss, _, _ = dino_loss_fn.calculate_reg_terms(z_eval_embeddings)
    print(f"Eval batch loss: {eval_loss.item():.4f}")
    print(f"Center after eval batch (should remain same): {dino_loss_fn.center[0, :3]}...")

    # Test input validation
    try:
        wrong_dim_z = torch.randn(batch_size, embed_dim + 1)
        dino_loss_fn(wrong_dim_z)
    except ValueError as e:
        print(f"\nSuccessfully caught error for wrong input dim: {e}")

    try:
        wrong_ndim_z = torch.randn(batch_size, embed_dim, 1)
        dino_loss_fn(wrong_ndim_z)
    except ValueError as e:
        print(f"\nSuccessfully caught error for wrong input ndim: {e}")
