import torch
import unittest
import copy # For deepcopying state_dicts
from src.models.jepa import JEPA

# Helper to initialize a JEPA model (vjepa2 mode for consistency, though EMA is general)
def get_jepa_model(device='cpu', input_channels=3, image_h_w=(64,64), patch_size=16, latent_dim=128, ema_decay=0.5): # Use a noticeable ema_decay for tests
    model = JEPA(
        image_size=image_h_w,
        patch_size=patch_size,
        input_channels=input_channels,
        action_dim=4,
        action_emb_dim=32,
        latent_dim=latent_dim,
        predictor_hidden_dim=256,
        predictor_output_dim=latent_dim,
        ema_decay=ema_decay, # Passed to model
        encoder_type='vit',
        encoder_params={'depth': 3, 'heads': 4, 'mlp_dim': 256, 'pool': 'cls'},
        target_encoder_mode="vjepa2" # EMA is active in vjepa2 and default
    ).to(device)
    return model

class TestJEPAEmaUpdate(unittest.TestCase):

    def test_target_encoder_ema_update(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ema_decay = 0.5 # Using a value that makes changes more obvious in a few steps
        model = get_jepa_model(device=device, ema_decay=ema_decay)

        # Ensure target_encoder exists
        self.assertIsNotNone(model.target_encoder, "Target encoder should be initialized.")
        # Ensure target encoder params initially don't require grad
        for param in model.target_encoder.parameters():
            self.assertFalse(param.requires_grad)


        # Store initial target_encoder parameters
        initial_target_params_state_dict = copy.deepcopy(model.target_encoder.state_dict())

        # --- Simulate a change in online_encoder parameters ---
        with torch.no_grad():
            for param_online in model.online_encoder.parameters():
                param_online.data.add_(torch.randn_like(param_online.data) * 0.1) # Add some noise

        # Store the state of the modified online_encoder
        modified_online_params_state_dict = copy.deepcopy(model.online_encoder.state_dict())

        # --- Call EMA update ---
        # The method in jepa.py is _update_target_encoder_ema, but it's called by perform_ema_update
        model.perform_ema_update()

        # --- Assertions ---
        updated_target_params_state_dict = model.target_encoder.state_dict()

        # Assertion 1: Target encoder parameters should be different from initial values
        changed_count = 0
        checked_one_param_formula = False
        for name, initial_param_val in initial_target_params_state_dict.items():
            updated_param_val = updated_target_params_state_dict[name]

            if not torch.equal(initial_param_val, updated_param_val):
                changed_count += 1

            # Check the formula for one parameter (e.g., the first one)
            if not checked_one_param_formula and name in modified_online_params_state_dict:
                expected_val = ema_decay * initial_param_val + (1 - ema_decay) * modified_online_params_state_dict[name]
                self.assertTrue(torch.allclose(updated_param_val, expected_val, atol=1e-6),
                                f"EMA update for param {name} does not match formula. Got {updated_param_val}, expected {expected_val}")
                checked_one_param_formula = True

        self.assertTrue(changed_count > 0, "Target encoder parameters should have changed after EMA update and online encoder modification.")
        self.assertTrue(checked_one_param_formula, "The EMA formula check was not performed for any parameter.")
        print(f"Number of changed parameter tensors in target encoder: {changed_count} / {len(initial_target_params_state_dict)}")


        # --- Perform multiple updates to see gradual approach ---
        print("Simulating multiple EMA updates with fixed online encoder...")

        # For this part, keep online_encoder fixed to see target approach it.
        # Online encoder is already at 'modified_online_params_state_dict' state.

        diff_norm_before_step_2 = 0
        with torch.no_grad():
            for name in modified_online_params_state_dict.keys():
                p_online = modified_online_params_state_dict[name]
                p_target = model.target_encoder.state_dict()[name] # Current target params
                diff_norm_before_step_2 += torch.norm(p_online - p_target).item()
        print(f"Norm of difference (Online - Target) before 2nd EMA call: {diff_norm_before_step_2:.6f}")

        model.perform_ema_update() # Second update

        diff_norm_after_step_2 = 0
        with torch.no_grad():
            for name in modified_online_params_state_dict.keys():
                p_online = modified_online_params_state_dict[name]
                p_target = model.target_encoder.state_dict()[name] # Current target params
                diff_norm_after_step_2 += torch.norm(p_online - p_target).item()
        print(f"Norm of difference (Online - Target) after 2nd EMA call: {diff_norm_after_step_2:.6f}")

        self.assertTrue(diff_norm_after_step_2 < diff_norm_before_step_2 - 1e-6, # check for strict decrease
                        f"Target encoder should be closer to online encoder after another EMA update if online is fixed. Diff_after: {diff_norm_after_step_2}, Diff_before: {diff_norm_before_step_2}")

        # Simulate one more change in online encoder and then EMA
        print("Simulating online encoder change and then 3rd EMA update...")
        with torch.no_grad():
            for param_online in model.online_encoder.parameters(): # Modifying the actual online_encoder parameters
                param_online.data.add_(torch.randn_like(param_online.data) * 0.1) # Add more noise
        # Update modified_online_params_state_dict to reflect this most recent change for accurate diff calculation
        current_online_params_state_dict = copy.deepcopy(model.online_encoder.state_dict())


        model.perform_ema_update() # Third update
        diff_norm_after_step_3 = 0
        with torch.no_grad():
            for name in current_online_params_state_dict.keys():
                p_online = current_online_params_state_dict[name]
                p_target = model.target_encoder.state_dict()[name]
                diff_norm_after_step_3 += torch.norm(p_online - p_target).item()
        print(f"Norm of difference (Online - Target) after 3rd EMA call (online changed again): {diff_norm_after_step_3:.6f}")
        # This new diff might be larger or smaller than diff_norm_after_step_2,
        # The important part is that EMA continues to work.
        # For example, it should be closer than if no EMA happened with the new online params.
        # ( (1-ema_decay) * norm_of_change_in_online_encoder ) vs ( norm_of_change_in_online_encoder )

if __name__ == '__main__':
    unittest.main()
