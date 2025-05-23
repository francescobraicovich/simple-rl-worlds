import torch
import torch.nn as nn
import torch.nn.functional as F

class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, eps=1e-4):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps # Epsilon for numerical stability in variance calculation

    def forward(self, x, y):
        # x, y: batches of embeddings, e.g., from two augmentations of the same input,
        # or in our JEPA case, x could be online_encoded_s_t and y could be online_encoded_s_t_plus_1
        # or we can apply it to a single set of embeddings (e.g., just online_encoded_s_t) by passing x=x.
        # For JEPA, we usually apply it to the batch of representations Z from the online encoder.
        # The original VICReg paper applies it to Z from two branches of augmentations.
        # Here, we'll make it flexible. If we want to apply to a single batch Z,
        # the calling code can pass Z as both x and y, and the invariance term (sim_loss) would be zero.
        # Or, more practically, we can have a method that just takes Z and computes var and cov terms.

        # Let's refine this: The VICReg paper applies it to X and Y, where X and Y are embeddings
        # of two differently augmented views of the same batch of images.
        # The loss is: L(X,Y) = lambda * sim(X,Y) + mu * var(X) + mu * var(Y) + nu * cov(X) + nu * cov(Y)
        # where var(X) is hinge_loss(std(X), gamma) and cov(X) is sum_off_diag(Cov(X)^2).

        # For JEPA, the "sim" part is the main prediction loss (predictor_output vs online_encoded_s_t_plus_1).
        # The "var" and "cov" parts are applied to the embeddings produced by the online encoder.
        # So, this loss function should primarily compute the variance and covariance penalties.
        # It can take one batch of embeddings Z and compute var(Z) and cov(Z).

        # Let's define two methods:
        # 1. `calculate_vic_terms(z)`: for a single batch of embeddings z.
        # 2. `forward(x, y)`: for the original two-branch VICReg (includes sim_loss).

        # For JEPA, we'll primarily use `calculate_vic_terms` on `online_encoded_s_t` and `online_encoded_s_t_plus_1`.

        # sim_loss: Invariance term (MSE between x and y)
        # This is the main JEPA prediction loss if x = predicted_embedding and y = target_embedding
        # However, the VICReg paper uses this for invariance between augmented views.
        # Let's stick to the original VICReg formulation for `forward(x,y)`.
        
        representation_loss = F.mse_loss(x, y)
        sim_loss = self.sim_coeff * representation_loss

        # Variance term for x
        x_std = torch.sqrt(x.var(dim=0) + self.eps) # (D,)
        std_loss_x = torch.mean(F.relu(1 - x_std)) # Hinge loss: max(0, 1 - std_dev)

        # Variance term for y
        y_std = torch.sqrt(y.var(dim=0) + self.eps) # (D,)
        std_loss_y = torch.mean(F.relu(1 - y_std)) # Hinge loss

        std_loss = self.std_coeff * (std_loss_x + std_loss_y) * 0.5 # Average over x and y

        # Covariance term for x
        # x is (N, D). Center it.
        x_centered = x - x.mean(dim=0)
        # Covariance matrix: (D, D)
        cov_x = (x_centered.T @ x_centered) / (x.size(0) - 1) 
        # Penalize off-diagonal elements
        # Sum of squared off-diagonal elements, divided by D to normalize
        cov_loss_x = (cov_x.fill_diagonal_(0).pow_(2).sum()) / x.size(1) # Division by D (num features)

        # Covariance term for y
        y_centered = y - y.mean(dim=0)
        cov_y = (y_centered.T @ y_centered) / (y.size(0) - 1)
        cov_loss_y = (cov_y.fill_diagonal_(0).pow_(2).sum()) / y.size(1)
        
        cov_loss = self.cov_coeff * (cov_loss_x + cov_loss_y) * 0.5 # Average over x and y

        total_loss = sim_loss + std_loss + cov_loss
        return total_loss, sim_loss, std_loss, cov_loss

    def calculate_reg_terms(self, z):
        # Calculates only variance and covariance regularization terms for a single batch of embeddings z.
        # This is what we'll likely use for JEPA's online encoder outputs.
        # z: (N, D) batch of embeddings

        # Variance term
        z_std = torch.sqrt(z.var(dim=0) + self.eps) # (D,)
        std_loss_val = torch.mean(F.relu(1 - z_std)) # Hinge loss

        # Covariance term
        z_centered = z - z.mean(dim=0)
        cov_z = (z_centered.T @ z_centered) / (z.size(0) - 1)
        cov_loss_val = (cov_z.fill_diagonal_(0).pow_(2).sum()) / z.size(1)

        weighted_std_loss = self.std_coeff * std_loss_val
        weighted_cov_loss = self.cov_coeff * cov_loss_val
        
        total_reg_loss = weighted_std_loss + weighted_cov_loss
        
        return total_reg_loss, weighted_std_loss, weighted_cov_loss

if __name__ == '__main__':
    # Example Usage
    vicreg = VICRegLoss(sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0)

    # For full VICReg (e.g., with augmentations)
    batch_size, embed_dim = 128, 256
    x_embeddings = torch.randn(batch_size, embed_dim)
    y_embeddings = x_embeddings + 0.05 * torch.randn_like(x_embeddings) # y is a slightly perturbed x

    total_loss_full, sim_l, std_l, cov_l = vicreg(x_embeddings, y_embeddings)
    print(f"Full VICReg -> Total: {total_loss_full.item():.4f}, Sim: {sim_l.item():.4f}, Std: {std_l.item():.4f}, Cov: {cov_l.item():.4f}")

    # For JEPA-style regularization (only std and cov terms on a single set of embeddings)
    z_embeddings = torch.randn(batch_size, embed_dim)
    # Simulate low variance for some features
    z_embeddings[:, :embed_dim//2] *= 0.1 
    # Simulate some covariance
    z_embeddings[:, 1] += 0.5 * z_embeddings[:, 0]

    total_reg, std_reg, cov_reg = vicreg.calculate_reg_terms(z_embeddings)
    print(f"Reg Terms Only (for JEPA) -> Total: {total_reg.item():.4f}, Std: {std_reg.item():.4f}, Cov: {cov_reg.item():.4f}")
    
    # Example with target std close to 1 and low covariance
    z_good_embeddings = torch.randn(batch_size, embed_dim)
    z_good_embeddings = F.normalize(z_good_embeddings, p=2, dim=1) # Normalize to unit sphere
    z_good_embeddings = z_good_embeddings * (embed_dim**0.5) # Scale to make std around 1 (Barlow Twins trick)
    # This normalization makes std of each sample 1. We need std of each feature over batch.
    # Let's try to make features have std=1 directly.
    z_good_embeddings = torch.randn(batch_size, embed_dim)
    z_good_embeddings = (z_good_embeddings - z_good_embeddings.mean(dim=0)) / (z_good_embeddings.std(dim=0) + 1e-5)


    total_reg_good, std_reg_good, cov_reg_good = vicreg.calculate_reg_terms(z_good_embeddings)
    print(f"Reg Terms (Good Embeddings) -> Total: {total_reg_good.item():.4f}, Std: {std_reg_good.item():.4f}, Cov: {cov_reg_good.item():.4f}")

```
