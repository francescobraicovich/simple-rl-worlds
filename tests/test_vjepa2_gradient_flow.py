import torch
import unittest
from src.models.jepa import JEPA
from src.losses.vicreg import VICRegLoss # Assuming VICRegLoss, adjust if different

# Helper to initialize a JEPA model in vjepa2 mode with basic params
def get_vjepa2_model(device='cpu', input_channels=3, image_h_w=(64,64), patch_size=16, latent_dim=128):
    # These are example parameters, might need adjustment based on actual model usage
    # Ensure encoder_params are correctly passed for the chosen encoder_type
    model = JEPA(
        image_size=image_h_w,
        patch_size=patch_size,
        input_channels=input_channels,
        action_dim=4, # Dummy action dim
        action_emb_dim=32, # Dummy action emb dim
        latent_dim=latent_dim,
        predictor_hidden_dim=256,
        predictor_output_dim=latent_dim, # Must match latent_dim
        ema_decay=0.996,
        encoder_type='vit', # Defaulting to ViT, ensure params match
        encoder_params={'depth': 3, 'heads': 4, 'mlp_dim': 256, 'pool': 'cls'}, # Minimal ViT params
        target_encoder_mode="vjepa2"
    ).to(device)
    return model

# Helper function to get VICRegLoss
def get_vicreg_loss(device='cpu'):
    return VICRegLoss(sim_coeff=1.0, std_coeff=1.0, cov_coeff=0.04).to(device) # Example coeffs

class TestVJEPA2GradientFlow(unittest.TestCase):

    def test_aux_loss_gradient_flow_to_online_encoder(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_vjepa2_model(device=device)
        aux_loss_fn = get_vicreg_loss(device=device)
        aux_loss_weight = 1.0 # Test with a significant weight

        # Ensure online_encoder parameters require gradients (should be default)
        for param in model.online_encoder.parameters():
            self.assertTrue(param.requires_grad)
        # Ensure predictor parameters require gradients (should be default)
        for param in model.predictor.parameters():
            self.assertTrue(param.requires_grad)
        # Ensure target_encoder parameters DO NOT require gradients
        if model.target_encoder:
            for param in model.target_encoder.parameters():
                self.assertFalse(param.requires_grad)

        # Get image_h_w and input_channels from the get_vjepa2_model defaults or arguments if changed
        # These are known values used to configure the model.
        config_input_channels = 3 # Default from get_vjepa2_model
        config_image_h_w = (64,64) # Default from get_vjepa2_model

        # Create dummy inputs
        batch_size = 2
        s_t = torch.randn(batch_size, config_input_channels, config_image_h_w[0], config_image_h_w[1], device=device)
        action_dim = 4 # Matching dummy action_dim in model init
        a_t = torch.rand(batch_size, action_dim, device=device) # Continuous actions for simplicity
        s_t_plus_1 = torch.randn(batch_size, config_input_channels, config_image_h_w[0], config_image_h_w[1], device=device)

        # Perform a forward pass
        # pred_emb, target_emb_detached, online_s_t_emb, online_s_t_plus_1_emb_vjepa2_is_None
        _, _, online_s_t_emb, _ = model(s_t, a_t, s_t_plus_1)

        # Calculate the auxiliary loss
        # Assuming calculate_reg_terms returns (loss, invariance_term, variance_term, covariance_term)
        # and we use the first component as the loss.
        loss_components = aux_loss_fn.calculate_reg_terms(online_s_t_emb)
        aux_loss = loss_components[0] * aux_loss_weight

        # Perform backward pass for aux_loss ONLY
        model.zero_grad() # Zero gradients before backward pass
        aux_loss.backward()

        # Assertion 1: Check that parameters of model.online_encoder have non-None gradients
        online_encoder_has_grads = False
        for param in model.online_encoder.parameters():
            if param.grad is not None:
                online_encoder_has_grads = True
                break
        self.assertTrue(online_encoder_has_grads, "Online encoder parameters should have gradients from auxiliary loss.")

        # Assertion 2 (Vanishing Gradient Check):
        total_grad_norm = 0.0
        num_params_with_grad = 0
        for name, param in model.online_encoder.named_parameters():
            if param.grad is not None:
                param_grad_norm = torch.linalg.norm(param.grad)
                total_grad_norm += param_grad_norm.item()
                num_params_with_grad +=1
                # print(f"Online Encoder Param: {name}, Grad Norm: {param_grad_norm.item()}") # For debugging
        self.assertTrue(num_params_with_grad > 0, "No parameters in online encoder received gradients.")
        avg_grad_norm = total_grad_norm / num_params_with_grad if num_params_with_grad > 0 else 0.0
        # This threshold is arbitrary and might need adjustment based on model scale/complexity
        self.assertTrue(avg_grad_norm > 1e-7, f"Average gradient norm for online_encoder ({avg_grad_norm}) is too small, possible vanishing/very small gradients.")
        print(f"Aux Loss Gradient Flow - Online Encoder Avg Grad Norm: {avg_grad_norm}")

        # Assertion 3: Verify that gradients are zero for model.predictor and model.target_encoder
        for name, param in model.predictor.named_parameters():
            self.assertIsNone(param.grad, f"Predictor parameter {name} should not have gradients from aux_loss applied only to online_s_t_emb.")

        if model.target_encoder: # Target encoder params should never have grads
            for name, param in model.target_encoder.named_parameters():
                self.assertIsNone(param.grad, f"Target encoder parameter {name} should not have gradients.")

if __name__ == '__main__':
    unittest.main()
