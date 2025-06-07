import unittest
import torch
import torch.nn as nn
import numpy as np

# Assuming the project structure allows these imports
# If run from project root:
from utils.data_utils import ExperienceDataset
from models.mlp import RewardPredictorMLP
# If torchvision is used by ExperienceDataset's default transform (it is)
from torchvision import transforms as T


class TestExperienceDatasetForRewards(unittest.TestCase):
    def test_yields_reward_tensor(self):
        print("\nRunning TestExperienceDatasetForRewards.test_yields_reward_tensor...")
        # Create dummy data
        num_samples = 10
        image_size = (3, 64, 64)  # C, H, W (after transform)

        # ExperienceDataset expects raw states (e.g., numpy HWC) to be transformed
        # Let's create dummy raw states (numpy arrays HWC)
        raw_image_h, raw_image_w, raw_channels = 64, 64, 3

        states_list = [np.random.randint(
            0, 256, (raw_image_h, raw_image_w, raw_channels), dtype=np.uint8) for _ in range(num_samples)]
        actions_list = [np.random.rand(1).astype(np.float32) for _ in range(
            num_samples)]  # Example continuous action
        rewards_list = [float(np.random.rand(1)) for _ in range(num_samples)]
        next_states_list = [np.random.randint(
            0, 256, (raw_image_h, raw_image_w, raw_channels), dtype=np.uint8) for _ in range(num_samples)]

        # Basic transform as used in data_utils
        preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize((image_size[1], image_size[2])),
            T.ToTensor()
        ])

        dataset = ExperienceDataset(
            states=states_list,
            actions=actions_list,
            rewards=rewards_list,
            next_states=next_states_list,
            transform=preprocess
        )

        self.assertTrue(len(dataset) > 0, "Dataset should not be empty")

        # Get an item
        s, a, r, ns = dataset[0]

        # Assertions for state s
        self.assertIsInstance(
            s, torch.Tensor, "State should be a torch.Tensor")
        self.assertEqual(
            s.shape, image_size, f"State shape incorrect, expected {image_size}, got {s.shape}")

        # Assertions for action a
        self.assertIsInstance(
            a, torch.Tensor, "Action should be a torch.Tensor")
        # Action dtype can vary, but data_utils casts to float32
        self.assertEqual(a.dtype, torch.float32,
                         "Action tensor dtype should be float32")

        # Assertions for reward r
        self.assertIsInstance(
            r, torch.Tensor, "Reward should be a torch.Tensor")
        self.assertEqual(r.dtype, torch.float32,
                         "Reward tensor dtype should be float32")
        # In __getitem__, reward is converted to a tensor but not unsqueezed.
        # It should be a scalar tensor (0-dim). In training, it's unsqueezed to (1,) or (batch, 1).
        self.assertEqual(
            r.ndim, 0, f"Reward tensor should be a scalar (0-dim), got {r.ndim}-dim, shape {r.shape}")

        # Assertions for next_state ns
        self.assertIsInstance(
            ns, torch.Tensor, "Next state should be a torch.Tensor")
        self.assertEqual(
            ns.shape, image_size, f"Next state shape incorrect, expected {image_size}, got {ns.shape}")

        print("TestExperienceDatasetForRewards.test_yields_reward_tensor completed successfully.")


class TestRewardPredictorMLP(unittest.TestCase):
    def test_forward_pass_shape(self):
        print("\nRunning TestRewardPredictorMLP.test_forward_pass_shape...")
        batch_size = 8

        test_configs = [
            {"input_dim": 128, "hidden_dims": [
                64, 32], "name": "Two hidden layers"},
            {"input_dim": 256, "hidden_dims": [],
                "name": "No hidden layers (linear)"},
            {"input_dim": 512, "hidden_dims": [
                256], "name": "One hidden layer"},
            {"input_dim": 128, "hidden_dims": [
                64, 32], "use_batch_norm": True, "name": "With Batch Norm"}
        ]

        for config in test_configs:
            print(f"  Testing RewardPredictorMLP with: {config['name']}")
            input_dim = config["input_dim"]
            hidden_dims = config["hidden_dims"]
            use_bn = config.get("use_batch_norm", False)

            mlp = RewardPredictorMLP(
                input_dim=input_dim, hidden_dims=hidden_dims, use_batch_norm=use_bn)
            if use_bn:  # Set to eval for consistent BN behavior if layers exist
                mlp.eval()

            dummy_input = torch.randn(batch_size, input_dim)
            output = mlp(dummy_input)

            expected_shape = (batch_size, 1)
            self.assertEqual(output.shape, expected_shape,
                             f"Output shape for '{config['name']}' mismatch. Expected {expected_shape}, got {output.shape}")
        print("TestRewardPredictorMLP.test_forward_pass_shape completed successfully.")

    def test_reward_mlp_training_step_shapes(self):
        print("\nRunning TestRewardPredictorMLP.test_reward_mlp_training_step_shapes...")
        batch_size = 16
        input_dim_mlp = 256  # Example input dimension for the MLP

        # Instantiate RewardPredictorMLP
        reward_mlp = RewardPredictorMLP(
            input_dim=input_dim_mlp, hidden_dims=[64, 32])
        reward_mlp.train()  # Set to train mode

        # Create dummy input for the MLP and dummy rewards
        dummy_mlp_input = torch.randn(batch_size, input_dim_mlp)
        dummy_rewards = torch.randn(batch_size, 1)  # Target rewards

        # Optimizer and loss function
        optimizer = torch.optim.Adam(reward_mlp.parameters(), lr=0.001)
        mse_loss = nn.MSELoss()

        # Training step
        optimizer.zero_grad()
        predicted_rewards = reward_mlp(dummy_mlp_input)

        # Assert shape of predicted_rewards
        self.assertEqual(predicted_rewards.shape, (batch_size, 1),
                         f"Predicted rewards shape mismatch. Expected {(batch_size, 1)}, got {predicted_rewards.shape}")

        loss = mse_loss(predicted_rewards, dummy_rewards)

        # Assert loss is a scalar tensor
        self.assertTrue(torch.is_tensor(loss), "Loss should be a tensor.")
        self.assertEqual(
            loss.ndim, 0, f"Loss should be a scalar tensor (0-dim), got {loss.ndim}-dim, shape {loss.shape}")

        loss.backward()
        optimizer.step()

        print("TestRewardPredictorMLP.test_reward_mlp_training_step_shapes completed successfully (shapes and flow).")


if __name__ == '__main__':
    unittest.main()
