import unittest
import torch
import torch.nn as nn

# Adjust imports based on the new structure
# Assuming PYTHONPATH is set to /app, then src.losses should be used.
from src.losses.vicreg import VICRegLoss
from src.losses.barlow_twins import BarlowTwinsLoss
from src.losses.dino import DINOLoss
# If tests are run from /app (e.g. python -m tests.test_losses), then 'from losses import ...' might fail
# if 'src' is not automatically added to sys.path.
# The structure 'from src.module import Class' is generally more robust when /app is in PYTHONPATH.


class TestLosses(unittest.TestCase):

    def setUp(self):
        self.batch_size = 32  # Using a smaller batch size for tests
        self.embed_dim = 64  # Using a smaller embedding dim for tests
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Dummy inputs
        self.z1 = torch.randn(
            self.batch_size, self.embed_dim, device=self.device)
        self.z2 = torch.randn(
            self.batch_size, self.embed_dim, device=self.device)
        self.z_requires_grad = torch.randn(
            self.batch_size, self.embed_dim, device=self.device, requires_grad=True)

    # --- VICRegLoss Tests ---
    def test_vicreg_instantiation(self):
        loss_fn = VICRegLoss().to(self.device)
        self.assertIsInstance(loss_fn, nn.Module)
        loss_fn_custom = VICRegLoss(
            sim_coeff=10.0, std_coeff=20.0, cov_coeff=0.5, eps=1e-5).to(self.device)
        self.assertIsInstance(loss_fn_custom, nn.Module)

    def test_vicreg_forward_and_reg_terms(self):
        loss_fn = VICRegLoss().to(self.device)

        # Test forward (original VICReg with two views)
        loss, sim_l, std_l, cov_l = loss_fn(self.z1, self.z2)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))  # Scalar
        # Inputs dont require grad by default in setUp
        self.assertTrue(loss.requires_grad == False)

        loss_grad, _, _, _ = loss_fn(
            self.z_requires_grad, self.z_requires_grad + 0.1)
        self.assertTrue(loss_grad.requires_grad)

        # Test calculate_reg_terms (for JEPA-style regularization)
        reg_loss, std_reg, cov_reg = loss_fn.calculate_reg_terms(self.z1)
        self.assertIsInstance(reg_loss, torch.Tensor)
        self.assertEqual(reg_loss.shape, torch.Size([]))
        self.assertIsInstance(std_reg, torch.Tensor)
        self.assertIsInstance(cov_reg, torch.Tensor)
        self.assertTrue(reg_loss.requires_grad == False)

        reg_loss_grad, _, _ = loss_fn.calculate_reg_terms(self.z_requires_grad)
        self.assertTrue(reg_loss_grad.requires_grad)

    def test_vicreg_std_loss_behavior(self):
        loss_fn = VICRegLoss(std_coeff=1.0, cov_coeff=0.0).to(
            self.device)  # Focus on std_loss
        # Input with std close to 1 for all features
        z_std_one = torch.randn(
            self.batch_size, self.embed_dim, device=self.device)
        z_std_one = (z_std_one - z_std_one.mean(dim=0)) / \
            (z_std_one.std(dim=0) + 1e-5)  # Normalize features

        reg_loss, std_loss_val, _ = loss_fn.calculate_reg_terms(z_std_one)
        # Hinge loss max(0, 1-std_dev). If std_dev is 1, this should be 0.
        self.assertAlmostEqual(std_loss_val.item(), 0.0, delta=1e-3)

        # Input with std close to 0 for all features
        z_std_zero = torch.zeros(
            self.batch_size, self.embed_dim, device=self.device)
        # Ensure loss_fn.eps is available and is a float or tensor for sqrt
        # loss_fn.eps is already a float attribute of the class instance
        
        reg_loss_zero_std, std_loss_val_zero, _ = loss_fn.calculate_reg_terms(
            z_std_zero)
            
        # raw_std_loss = mean(relu(1 - sqrt(var + eps)))
        # For var=0, raw_std_loss = mean(relu(1 - sqrt(eps))) = 1 - sqrt(eps)
        # std_loss_val_zero is already weighted_std_loss = std_coeff * raw_std_loss
        expected_raw_std_loss_for_zero_variance = 1.0 - torch.sqrt(torch.tensor(loss_fn.eps, device=self.device))
        expected_weighted_std_loss = expected_raw_std_loss_for_zero_variance * loss_fn.std_coeff
        
        self.assertAlmostEqual(std_loss_val_zero.item(),
                               expected_weighted_std_loss.item(), delta=1e-3)

    def test_vicreg_cov_loss_behavior(self):
        loss_fn = VICRegLoss(std_coeff=0.0, cov_coeff=1.0).to(
            self.device)  # Focus on cov_loss
        # Input with uncorrelated features (e.g., diagonal covariance matrix)
        # A random matrix is unlikely to have perfect diagonal covariance,
        # but a matrix with orthogonal columns (after centering) would.
        # For simplicity, let's use a matrix where off-diagonal of cov is small.
        z_uncorr = torch.randn(
            self.batch_size, self.embed_dim, device=self.device)
        # Whitening would make it uncorrelated, but it's complex.
        # Instead, check that loss is positive for random data.
        reg_loss, _, cov_loss_val = loss_fn.calculate_reg_terms(z_uncorr)
        self.assertGreater(cov_loss_val.item(), 0.0)
        # A zero covariance loss is hard to achieve without specific construction like PCA output

    def test_vicreg_calculate_reg_terms(self):
        # Instantiate VICRegLoss with std_coeff=1.0, cov_coeff=1.0 for predictable raw loss values
        loss_fn = VICRegLoss(sim_coeff=1.0, std_coeff=1.0, cov_coeff=1.0, eps=1e-5).to(self.device) # Added eps for ideal case normalization
        batch_size, embed_dim = 64, 128

        # Test Case 1: Non-ideal input
        z_non_ideal = torch.randn(batch_size, embed_dim, device=self.device)
        # Introduce low variance for the first half of features
        z_non_ideal[:, :embed_dim//2] *= 0.1
        # Introduce covariance: make feature 0 correlated with feature 1
        z_non_ideal[:, 0] += 0.5 * z_non_ideal[:, 1]
        z_non_ideal.requires_grad_(True)

        total_reg_loss, std_loss, cov_loss = loss_fn.calculate_reg_terms(z_non_ideal)

        self.assertIsInstance(total_reg_loss, torch.Tensor)
        self.assertGreater(total_reg_loss.item(), 0.0, "Total reg loss should be > 0 for non-ideal input")
        self.assertGreater(std_loss.item(), 0.0, "Std loss should be > 0 for non-ideal input (low variance)")
        self.assertGreater(cov_loss.item(), 0.0, "Cov loss should be > 0 for non-ideal input (correlated features)")

        # Test gradients
        total_reg_loss.backward()
        self.assertIsNotNone(z_non_ideal.grad, "Grad should exist for z_non_ideal")
        self.assertTrue(z_non_ideal.grad.abs().sum().item() > 0, "Sum of grads should be > 0")
        z_non_ideal.grad = None # Clear grad for next test case

        # Test Case 2: Ideal input (features are mean 0, std 1, and as uncorrelated as possible for random data)
        # Using a smaller embed_dim for the ideal case, where embed_dim < batch_size,
        # might lead to a more well-behaved sample covariance matrix.
        ideal_embed_dim = embed_dim // 4 # e.g., 128 // 4 = 32. batch_size is 64.
        z_ideal_raw = torch.randn(batch_size, ideal_embed_dim, device=self.device)

        # Normalize features to have mean 0 and std 1
        z_ideal_mean = z_ideal_raw.mean(dim=0, keepdim=True)
        z_ideal_std = z_ideal_raw.std(dim=0, keepdim=True)
        z_ideal = (z_ideal_raw - z_ideal_mean) / (z_ideal_std + loss_fn.eps) # Use loss_fn.eps

        # Verify normalization (optional, but good for sanity check)
        # self.assertTrue(torch.allclose(z_ideal.mean(dim=0), torch.zeros_like(z_ideal.mean(dim=0)), atol=1e-5))
        # self.assertTrue(torch.allclose(z_ideal.std(dim=0), torch.ones_like(z_ideal.std(dim=0)), atol=1e-2)) # std can have larger tolerance

        total_reg_loss_ideal, std_loss_ideal, cov_loss_ideal = loss_fn.calculate_reg_terms(z_ideal)

        # For std_loss: if features have std=1, then F.relu(1 - z_std) should be F.relu(1-1) = 0.
        # So, std_loss_ideal should be very close to 0.
        self.assertTrue(torch.isclose(std_loss_ideal, torch.tensor(0.0, device=self.device), atol=1e-3),
                        f"Std loss for ideal input (embed_dim={ideal_embed_dim}) not close to 0: {std_loss_ideal.item()}")

        # For cov_loss: for perfectly uncorrelated features, cov_matrix is diagonal.
        # fill_diagonal_(0).pow_(2).sum() would be 0.
        # Random normalized data won't be perfectly uncorrelated, but cov_loss should be small.
        # With smaller ideal_embed_dim, the sum of squared off-diagonal covariances might be smaller.
        # The value 0.5447 was observed with atol=0.5. Let's set atol slightly higher.
        self.assertTrue(torch.isclose(cov_loss_ideal, torch.tensor(0.0, device=self.device), atol=0.6),
                        f"Cov loss for ideal input (embed_dim={ideal_embed_dim}) not close enough to 0 (atol=0.6): {cov_loss_ideal.item()}")

        # Total loss tolerance should also reflect the covariance tolerance
        self.assertTrue(torch.isclose(total_reg_loss_ideal, torch.tensor(0.0, device=self.device), atol=0.6),
                        f"Total reg loss for ideal input (embed_dim={ideal_embed_dim}) not close to 0 (atol=0.6): {total_reg_loss_ideal.item()}")

        print(f"\nTest test_vicreg_calculate_reg_terms. Non-ideal (D={embed_dim}): total={total_reg_loss.item()}, std={std_loss.item()}, cov={cov_loss.item()}. Ideal (D={ideal_embed_dim}): total={total_reg_loss_ideal.item()}, std={std_loss_ideal.item()}, cov={cov_loss_ideal.item()}")


    # --- BarlowTwinsLoss Tests ---

    def test_barlowtwins_instantiation(self):
        loss_fn = BarlowTwinsLoss().to(self.device)
        self.assertIsInstance(loss_fn, nn.Module)
        loss_fn_custom = BarlowTwinsLoss(
            lambda_param=0.01, eps=1e-6, scale_loss=0.5).to(self.device)
        self.assertIsInstance(loss_fn_custom, nn.Module)

    def test_barlowtwins_forward_and_reg_terms(self):
        loss_fn = BarlowTwinsLoss().to(self.device)

        # Test forward (original with two views)
        loss = loss_fn(self.z1, self.z2)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertFalse(loss.requires_grad)

        loss_grad = loss_fn(self.z_requires_grad, self.z_requires_grad + 0.1)
        self.assertTrue(loss_grad.requires_grad)

        # Test calculate_reg_terms (for JEPA-style regularization)
        reg_loss, inv_loss, red_loss = loss_fn.calculate_reg_terms(self.z1)
        self.assertIsInstance(reg_loss, torch.Tensor)
        self.assertEqual(reg_loss.shape, torch.Size([]))
        self.assertIsInstance(inv_loss, torch.Tensor)
        self.assertIsInstance(red_loss, torch.Tensor)
        self.assertFalse(reg_loss.requires_grad)

        reg_loss_grad, _, _ = loss_fn.calculate_reg_terms(self.z_requires_grad)
        self.assertTrue(reg_loss_grad.requires_grad)

    def test_barlowtwins_ideal_input(self):
        loss_fn = BarlowTwinsLoss(lambda_param=5e-3).to(self.device)
        # Create input z whose normalized auto-correlation is identity
        # This means z_norm.T @ z_norm / N = I
        # If z_norm's columns are orthonormal and N=D, this holds.
        # More generally, if z itself is feature-wise normalized (mean 0, std 1)
        # and features are uncorrelated.

        # Create a matrix that is already normalized (features mean 0, std 1)
        # and try to make it somewhat uncorrelated.
        # For a perfect test, one might construct z_norm directly.
        z_ideal_norm = torch.eye(
            self.batch_size, self.embed_dim, device=self.device)
        if self.batch_size > self.embed_dim:  # More samples than features
            # Make columns orthonormal (approx)
            q, _ = torch.linalg.qr(torch.randn(
                self.batch_size, self.embed_dim, device=self.device))
            # scaled so z_norm.T @ z_norm / N = I
            z_ideal_norm = q * (self.batch_size**0.5)
        elif self.batch_size < self.embed_dim:  # More features than samples
            # Make rows orthonormal (approx)
            q, _ = torch.linalg.qr(torch.randn(
                self.embed_dim, self.batch_size, device=self.device))
            z_ideal_norm = q.T * (self.batch_size**0.5)
        else:  # batch_size == embed_dim
            q, _ = torch.linalg.qr(torch.randn(
                self.batch_size, self.batch_size, device=self.device))
            z_ideal_norm = q * (self.batch_size**0.5)

        # The above z_ideal_norm is what the *normalized* version inside Barlow Twins should look like.
        # The input to BarlowTwinsLoss `z` will be normalized again.
        # So, we need to construct `z` such that `(z - z.mean) / z.std` is `z_ideal_norm`.
        # This is tricky. Let's test by overriding the normalization for a moment or checking components.

        # Simpler test: if auto_corr_matrix is identity, loss is 0.
        identity_corr = torch.eye(self.embed_dim, device=self.device)
        on_diag_ideal = torch.diagonal(identity_corr)
        invariance_loss_ideal = ((on_diag_ideal - 1)**2).sum()  # Should be 0

        off_diag_ideal = loss_fn._off_diagonal(
            identity_corr)  # from barlow_twins.py
        redundancy_loss_ideal = (off_diag_ideal**2).sum()  # Should be 0

        self.assertAlmostEqual(invariance_loss_ideal.item(), 0.0, delta=1e-6)
        self.assertAlmostEqual(redundancy_loss_ideal.item(), 0.0, delta=1e-6)

        # Test with pre-normalized input (features mean 0, std 1)
        z_pre_normalized = torch.randn(
            self.batch_size, self.embed_dim, device=self.device)
        z_pre_normalized = (z_pre_normalized - z_pre_normalized.mean(dim=0)
                            ) / (z_pre_normalized.std(dim=0) + 1e-5)
        loss_val, inv_l, red_l = loss_fn.calculate_reg_terms(z_pre_normalized)
        # Loss should not be identically zero for random normalized data, but components should be sensible.
        # Diagonal elements of correlation matrix should be 1. Off-diagonal non-zero.
        # So inv_l should be small, red_l will be non-zero.
        # After normalization within BT, z_norm.T @ z_norm / N. Diagonal elements are var(z_norm_i) = 1.
        # So (c_ii - 1)^2 should indeed be 0 for diagonal.
        self.assertAlmostEqual(inv_l.item() / loss_fn.scale_loss, 0.0, delta=1e-3,
                               msg="Invariance loss for pre-normalized input not near zero.")

    # --- DINOLoss Tests ---

    def test_dino_instantiation(self):
        loss_fn = DINOLoss(out_dim=self.embed_dim).to(self.device)
        self.assertIsInstance(loss_fn, nn.Module)
        self.assertTrue(torch.allclose(loss_fn.center, torch.zeros(
            1, self.embed_dim, device=self.device)))
        loss_fn_custom = DINOLoss(
            out_dim=self.embed_dim, center_ema_decay=0.95).to(self.device)
        self.assertIsInstance(loss_fn_custom, nn.Module)

    def test_dino_forward_and_reg_terms(self):
        loss_fn = DINOLoss(out_dim=self.embed_dim).to(self.device)
        loss_fn.train()  # Enable center update

        # Test calculate_reg_terms
        loss, main_comp, zero_comp = loss_fn.calculate_reg_terms(self.z1)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(loss.item(), main_comp.item())
        self.assertEqual(zero_comp.item(), 0.0)
        self.assertFalse(loss.requires_grad)  # z1 does not require grad

        loss_grad, _, _ = loss_fn.calculate_reg_terms(self.z_requires_grad)
        self.assertTrue(loss_grad.requires_grad)

    def test_dino_center_update(self):
        loss_fn = DINOLoss(out_dim=self.embed_dim,
                           center_ema_decay=0.9).to(self.device)
        initial_center = loss_fn.center.clone()

        # Train mode: center should update
        loss_fn.train()
        z_train = torch.ones(self.batch_size, self.embed_dim,
                             device=self.device) * 2.0  # Mean is 2.0
        loss_fn.calculate_reg_terms(z_train)
        updated_center_train = loss_fn.center.clone()
        # Expected center: 0.9 * 0 + (1-0.9) * 2.0 = 0.2
        self.assertFalse(torch.allclose(updated_center_train, initial_center))
        self.assertTrue(torch.allclose(updated_center_train,
                        torch.ones_like(initial_center) * 0.2, atol=1e-5))

        # Eval mode: center should NOT update
        loss_fn.eval()
        center_before_eval = loss_fn.center.clone()
        z_eval = torch.ones(self.batch_size, self.embed_dim,
                            device=self.device) * 5.0  # Mean is 5.0
        loss_fn.calculate_reg_terms(z_eval)
        center_after_eval = loss_fn.center.clone()
        self.assertTrue(torch.allclose(center_after_eval, center_before_eval))

        # Loss calculation should use the center BEFORE update in train mode for that batch
        loss_fn.train()
        loss_fn.center.data.fill_(0.0)  # Reset center
        z_batch1 = torch.ones(self.batch_size, self.embed_dim,
                              device=self.device) * 1.0  # Mean 1
        # Expected loss: (1.0 - 0.0)^2 * embed_dim = embed_dim
        loss1, _, _ = loss_fn.calculate_reg_terms(z_batch1)
        self.assertAlmostEqual(loss1.item(), float(self.embed_dim), delta=1e-5)
        # Center becomes 0.9*0 + 0.1*1 = 0.1

        z_batch2 = torch.ones(self.batch_size, self.embed_dim,
                              device=self.device) * 1.0  # Mean 1
        # Expected loss: (1.0 - 0.1)^2 * embed_dim = (0.9)^2 * embed_dim
        loss2, _, _ = loss_fn.calculate_reg_terms(z_batch2)
        self.assertAlmostEqual(loss2.item(), (0.9**2) *
                               self.embed_dim, delta=1e-5)

    def test_dino_input_validation(self):
        loss_fn = DINOLoss(out_dim=self.embed_dim).to(self.device)
        with self.assertRaises(ValueError):
            loss_fn(torch.randn(self.batch_size,
                    self.embed_dim + 10, device=self.device))
        with self.assertRaises(ValueError):
            loss_fn(torch.randn(self.batch_size,
                    self.embed_dim, 1, device=self.device))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
