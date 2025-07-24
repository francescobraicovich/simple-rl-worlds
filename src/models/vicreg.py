"""
VICReg Loss Implementation with Optional Projector Head

This implementation follows the original VICReg paper approach with an optional projector head
for better decoupling between representation learning and regularization terms.

Key features:
- Invariance term: MSE loss between projected representations
- Variance term: Prevents dimensional collapse using hinge loss
- Covariance term: Decorrelates features to prevent redundancy
- Optional projector: 3-layer MLP with BatchNorm and ReLU for better decoupling

Usage:
- Without projector: VICRegLoss() - operates directly on input representations
- With projector: VICRegLoss(representation_dim=dim) - adds projector for better decoupling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff=1.0, std_coeff=1.0, cov_coeff=1.0, eps=1e-4, 
                 proj_hidden_dim=8192, proj_output_dim=8192, representation_dim=None):
        super().__init__()

        print('Initialising with parameters:')
        print(f'sim_coeff: {sim_coeff}, std_coeff: {std_coeff}, cov_coeff: {cov_coeff}, eps: {eps}')
        print(f'proj_hidden_dim: {proj_hidden_dim}, proj_output_dim: {proj_output_dim}, representation_dim: {representation_dim}')
        
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps  # Epsilon for numerical stability in variance calculation
        
        # Projector head for better decoupling (as in original VICReg)
        if representation_dim is not None:
            self.projector = nn.Sequential(
                nn.Linear(representation_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_hidden_dim, proj_output_dim)
            )
        else:
            self.projector = None

    def forward(self, x, y):
        # Apply projector if available (for better decoupling)
        x = x.squeeze(1)
        y = y.squeeze(1)

        assert x.shape == y.shape, f'Input shapes do not match: {x.shape} vs {y.shape}'

        if self.projector is not None:
            x_proj = self.projector(x)
            y_proj = self.projector(y)
        else:
            x_proj, y_proj = x, y
        
        # Clamp projections to prevent extreme values
        x_proj = torch.clamp(x_proj, -10, 10)
        y_proj = torch.clamp(y_proj, -10, 10)
        
        # Calculate similarity loss (invariance term)
        representation_loss = F.mse_loss(x_proj, y_proj)
        sim_loss = self.sim_coeff * representation_loss

        # Calculate variance loss (prevents collapse) with numerical stability
        x_std = torch.sqrt(torch.clamp(x_proj.var(dim=0), min=self.eps))  # (D,)
        # Hinge loss: max(0, 1 - std_dev)
        std_loss_x = torch.mean(F.relu(1 - x_std))

        y_std = torch.sqrt(torch.clamp(y_proj.var(dim=0), min=self.eps))  # (D,)
        std_loss_y = torch.mean(F.relu(1 - y_std))  # Hinge loss

        std_loss = self.std_coeff * (std_loss_x + std_loss_y) * 0.5  # Average over x and y

        # Calculate covariance loss (prevents redundant features) with numerical stability
        x_centered = x_proj - x_proj.mean(dim=0)
        cov_x = (x_centered.T @ x_centered) / max(x_proj.size(0) - 1, 1)
        # Clamp covariance matrix to prevent explosions
        cov_x = torch.clamp(cov_x, -100, 100)
        cov_loss_x = (cov_x.fill_diagonal_(0).pow_(2).sum()) / \
            x_proj.size(1)  # Division by D (num features)

        y_centered = y_proj - y_proj.mean(dim=0)
        cov_y = (y_centered.T @ y_centered) / max(y_proj.size(0) - 1, 1)
        # Clamp covariance matrix to prevent explosions
        cov_y = torch.clamp(cov_y, -100, 100)
        cov_loss_y = (cov_y.fill_diagonal_(0).pow_(2).sum()) / y_proj.size(1)

        cov_loss = self.cov_coeff * (cov_loss_x + cov_loss_y) * 0.5  # Average over x and y

        # Clamp individual loss components to prevent explosions
        sim_loss = torch.clamp(sim_loss, 0, 100)
        std_loss = torch.clamp(std_loss, 0, 100)
        cov_loss = torch.clamp(cov_loss, 0, 100)

        total_loss = sim_loss + std_loss + cov_loss
        return total_loss, sim_loss, std_loss, cov_loss

    def calculate_reg_terms(self, z):
        # Apply projector if available (for better decoupling)
        if self.projector is not None:
            z_proj = self.projector(z)
        else:
            z_proj = z
        
        # Clamp projections to prevent extreme values
        z_proj = torch.clamp(z_proj, -10, 10)
            
        z_std = torch.sqrt(torch.clamp(z_proj.var(dim=0), min=self.eps))  # (D,)
        std_loss_val = torch.mean(F.relu(1 - z_std))  # Hinge loss

        z_centered = z_proj - z_proj.mean(dim=0)
        cov_z = (z_centered.T @ z_centered) / max(z_proj.size(0) - 1, 1)
        
        # Clamp covariance matrix to prevent explosions
        cov_z = torch.clamp(cov_z, -100, 100)
        cov_loss_val = (cov_z.fill_diagonal_(0).pow_(2).sum()) / z_proj.size(1)

        # Check for extreme values and warn
        if cov_loss_val > 1e6:
            print(f'Warning: cov_loss_val is extremely high: {cov_loss_val:.2e}')
            print(f'z_proj stats - mean: {z_proj.mean():.4f}, std: {z_proj.std():.4f}')
            print(f'cov_z max abs value: {cov_z.abs().max():.4f}')
            # Emergency clamp
            cov_loss_val = torch.clamp(cov_loss_val, 0, 1e6)

        # Clamp loss components to prevent explosions
        std_loss_val = torch.clamp(std_loss_val, 0, 100)
        cov_loss_val = torch.clamp(cov_loss_val, 0, 100)

        weighted_std_loss = self.std_coeff * std_loss_val
        weighted_cov_loss = self.cov_coeff * cov_loss_val

        total_reg_loss = weighted_std_loss + weighted_cov_loss
        return total_reg_loss, weighted_std_loss, weighted_cov_loss
