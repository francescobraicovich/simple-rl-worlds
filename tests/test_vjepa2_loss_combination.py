import torch
import torch.nn as nn
import unittest
from src.models.jepa import JEPA
from src.losses.vicreg import VICRegLoss # Assuming VICRegLoss

# Helper to initialize a JEPA model in vjepa2 mode
def get_vjepa2_model(device='cpu', input_channels=3, image_h_w=(64,64), patch_size=16, latent_dim=128):
    model = JEPA(
        image_size=image_h_w,
        patch_size=patch_size,
        input_channels=input_channels,
        action_dim=4,
        action_emb_dim=32,
        latent_dim=latent_dim,
        predictor_hidden_dim=256,
        predictor_output_dim=latent_dim,
        ema_decay=0.996,
        encoder_type='vit',
        encoder_params={'depth': 3, 'heads': 4, 'mlp_dim': 256, 'pool': 'cls'},
        target_encoder_mode="vjepa2"
    ).to(device)
    return model

# Helper function to get VICRegLoss
def get_vicreg_loss(device='cpu'):
    return VICRegLoss(sim_coeff=1.0, std_coeff=1.0, cov_coeff=0.04).to(device)

# Helper to get gradients from parameters
def get_grad_dict(model_component):
    grad_dict = {}
    for name, param in model_component.named_parameters():
        if param.grad is not None:
            grad_dict[name] = param.grad.clone()
        else:
            grad_dict[name] = None
    return grad_dict

# Helper to calculate total norm of a grad_dict
def calculate_total_norm(grad_dict):
    total_norm_sq = 0.0
    num_grads = 0
    for name, grad_tensor in grad_dict.items():
        if grad_tensor is not None:
            total_norm_sq += grad_tensor.norm().pow(2).item()
            num_grads +=1
    return torch.sqrt(torch.tensor(total_norm_sq)).item() if num_grads > 0 else 0.0

class TestVJEPA2LossCombination(unittest.TestCase):

    def test_loss_combination_gradient_impact(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Using fixed config values consistent with get_vjepa2_model defaults
        config_input_channels = 3
        config_image_h_w = (64,64)

        model = get_vjepa2_model(device=device, input_channels=config_input_channels, image_h_w=config_image_h_w)

        # Ensure correct parts of the model require/don't require grads initially
        for param in model.online_encoder.parameters(): param.requires_grad_(True)
        for param in model.predictor.parameters(): param.requires_grad_(True)
        if model.target_encoder:
            for param in model.target_encoder.parameters(): param.requires_grad_(False)

        aux_loss_fn = get_vicreg_loss(device=device)
        main_loss_fn = nn.MSELoss()

        # Dummy inputs
        batch_size = 2
        s_t = torch.randn(batch_size, config_input_channels, config_image_h_w[0], config_image_h_w[1], device=device)
        action_dim = 4 # Matching dummy action_dim in model init
        a_t = torch.rand(batch_size, action_dim, device=device)
        s_t_plus_1 = torch.randn(batch_size, config_input_channels, config_image_h_w[0], config_image_h_w[1], device=device)

        # --- 1. Gradients from Main Loss Only ---
        # Forward pass
        pred_emb_main, target_emb_detached_main, _, _ = model(s_t, a_t, s_t_plus_1)
        main_loss_val = main_loss_fn(pred_emb_main, target_emb_detached_main)

        self.assertTrue(main_loss_val.item() > 0, "Main loss should be positive.")

        model.zero_grad()
        main_loss_val.backward(retain_graph=True)
        grads_main_loss_online_encoder = get_grad_dict(model.online_encoder)
        grads_main_loss_predictor = get_grad_dict(model.predictor)
        norm_main_online_encoder = calculate_total_norm(grads_main_loss_online_encoder)
        norm_main_predictor = calculate_total_norm(grads_main_loss_predictor)
        print(f"Norm of Grads (Main Loss Only) - Online Encoder: {norm_main_online_encoder}, Predictor: {norm_main_predictor}")
        self.assertTrue(norm_main_online_encoder > 1e-7, "Online encoder grads from main loss are too small.")
        self.assertTrue(norm_main_predictor > 1e-7, "Predictor grads from main loss are too small.")

        # --- 2. Gradients from Aux Loss Only (representative weight 1.0) ---
        # Forward pass
        _, _, online_s_t_emb_aux, _ = model(s_t, a_t, s_t_plus_1)
        aux_loss_components = aux_loss_fn.calculate_reg_terms(online_s_t_emb_aux)
        aux_loss_val_scalar = aux_loss_components[0]

        self.assertTrue(aux_loss_val_scalar.item() > 0, "Aux loss should be positive for non-identical inputs.")

        model.zero_grad()
        (aux_loss_val_scalar * 1.0).backward(retain_graph=True)
        grads_aux_loss_online_encoder = get_grad_dict(model.online_encoder)
        grads_aux_loss_predictor = get_grad_dict(model.predictor)
        norm_aux_online_encoder = calculate_total_norm(grads_aux_loss_online_encoder)
        norm_aux_predictor = calculate_total_norm(grads_aux_loss_predictor)
        print(f"Norm of Grads (Aux Loss Only, w=1.0) - Online Encoder: {norm_aux_online_encoder}, Predictor: {norm_aux_predictor}")
        self.assertTrue(norm_aux_online_encoder > 1e-7, "Online encoder grads from aux loss are too small.")
        # When backpropagating *only* the aux_loss (calculated from online_s_t_emb),
        # the predictor should not receive gradients from this specific loss term.
        self.assertAlmostEqual(norm_aux_predictor, 0.0, delta=1e-7, msg="Predictor grads from aux loss only should be zero or very close to zero.")


        # --- 3. Gradients from Combined Loss (Low Aux Weight) ---
        pred_emb_low, target_emb_detached_low, online_s_t_emb_low, _ = model(s_t, a_t, s_t_plus_1)
        main_loss_low = main_loss_fn(pred_emb_low, target_emb_detached_low)
        aux_loss_scalar_low = aux_loss_fn.calculate_reg_terms(online_s_t_emb_low)[0]
        total_loss_low_weight = main_loss_low + aux_loss_scalar_low * 0.001

        model.zero_grad()
        total_loss_low_weight.backward(retain_graph=True)
        grads_total_low_online_encoder = get_grad_dict(model.online_encoder)
        norm_total_low_online_encoder = calculate_total_norm(grads_total_low_online_encoder)
        print(f"Norm of Grads (Combined, Low Aux Weight) - Online Encoder: {norm_total_low_online_encoder}")

        self.assertAlmostEqual(norm_total_low_online_encoder, norm_main_online_encoder, delta=norm_main_online_encoder*0.1,
                               msg="Grad norm with low aux weight should be close to main loss only grad norm.")

        # --- 4. Gradients from Combined Loss (High Aux Weight) ---
        pred_emb_high, target_emb_detached_high, online_s_t_emb_high, _ = model(s_t, a_t, s_t_plus_1)
        main_loss_high = main_loss_fn(pred_emb_high, target_emb_detached_high)
        aux_loss_scalar_high = aux_loss_fn.calculate_reg_terms(online_s_t_emb_high)[0]
        total_loss_high_weight = main_loss_high + aux_loss_scalar_high * 1.0

        model.zero_grad()
        total_loss_high_weight.backward()
        grads_total_high_online_encoder = get_grad_dict(model.online_encoder)
        norm_total_high_online_encoder = calculate_total_norm(grads_total_high_online_encoder)
        print(f"Norm of Grads (Combined, High Aux Weight) - Online Encoder: {norm_total_high_online_encoder}")

        self.assertNotAlmostEqual(norm_total_high_online_encoder, norm_main_online_encoder, delta=norm_main_online_encoder*0.05,
                                  msg="Grad norm with high aux weight should differ from main loss only grad norm.")

        # Check that the combined norm is different from aux_loss_only norm,
        # using a delta relative to the main_loss_norm, as main_loss is the smaller component here.
        # This ensures main_loss has made a non-negligible contribution.
        self.assertNotAlmostEqual(norm_total_high_online_encoder, norm_aux_online_encoder, delta=norm_main_online_encoder*0.1, # Delta based on main_loss_norm
                                  msg="Grad norm with high aux weight should differ from aux loss only grad norm, indicating main loss contribution.")

if __name__ == '__main__':
    unittest.main()
