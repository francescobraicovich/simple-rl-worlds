from utils.data_utils import collect_random_episodes, ExperienceDataset
import unittest
import os
import sys
import gymnasium as gym
import math  # For ceiling function in episode count verification

# Add project root to sys.path to allow importing project modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)


# A simple environment for testing
ENV_NAME = "CartPole-v1"
try:
    gym.make(ENV_NAME).close()
    ENV_AVAILABLE = True
except Exception:
    ENV_AVAILABLE = False


@unittest.skipIf(not ENV_AVAILABLE, f"Environment {ENV_NAME} not available. Skipping data_utils tests.")
class TestDataUtils(unittest.TestCase):

    def test_collect_random_episodes_split(self):
        num_episodes_total = 10
        max_steps_per_episode = 5  # Each episode will have at most this many transitions
        # CartPole doesn't use image_size, but the function requires it.
        # It's used by the internal transform. Let's use a dummy one.
        # The actual observation from CartPole is a 4-element array, not an image.
        # The current data_utils.py tries to force it into an image-like pipeline
        # which might cause issues for non-image envs if not handled carefully.
        # For this test, we assume collect_random_episodes can run with CartPole-v1
        # by correctly handling its observations (e.g. if it renders them).
        # The `preprocess` in data_utils uses ToPILImage, Resize, ToTensor.
        # env.render() for CartPole might return an image.
        image_size = (64, 64)

        print(
            f"\nStarting test_collect_random_episodes_split with {ENV_NAME}...")
        print(
            f"Total episodes for tests: {num_episodes_total}, max_steps: {max_steps_per_episode}")

        # Test Case 1: 20% validation split
        val_split_ratio_1 = 0.2
        # num_val_episodes is int(validation_split_ratio * num_total_episodes)
        expected_val_episodes_1 = int(
            num_episodes_total * val_split_ratio_1)  # 10 * 0.2 = 2
        expected_train_episodes_1 = num_episodes_total - \
            expected_val_episodes_1  # 10 - 2 = 8

        print(f"\nTest Case 1: val_split_ratio={val_split_ratio_1}")
        print(
            f"Expected train episodes: {expected_train_episodes_1}, val episodes: {expected_val_episodes_1}")
        train_dataset_1, val_dataset_1 = collect_random_episodes(
            ENV_NAME, num_episodes_total, max_steps_per_episode, image_size, val_split_ratio_1
        )

        self.assertIsInstance(train_dataset_1, ExperienceDataset)
        self.assertIsInstance(val_dataset_1, ExperienceDataset)

        # Check number of transitions. It should be roughly proportional to episodes.
        # All episodes should contain between 1 and max_steps_per_episode transitions.
        if expected_train_episodes_1 > 0:
            self.assertTrue(len(train_dataset_1) >= expected_train_episodes_1 * 1,
                            f"Train dataset 1 has {len(train_dataset_1)} transitions, expected at least {expected_train_episodes_1}")
            self.assertTrue(len(train_dataset_1) <= expected_train_episodes_1 * max_steps_per_episode,
                            f"Train dataset 1 has {len(train_dataset_1)} transitions, expected at most {expected_train_episodes_1 * max_steps_per_episode}")
        else:
            self.assertEqual(len(train_dataset_1), 0,
                             "Train dataset 1 should be empty")

        if expected_val_episodes_1 > 0:
            self.assertTrue(len(val_dataset_1) >= expected_val_episodes_1 * 1,
                            f"Validation dataset 1 has {len(val_dataset_1)} transitions, expected at least {expected_val_episodes_1}")
            self.assertTrue(len(val_dataset_1) <= expected_val_episodes_1 * max_steps_per_episode,
                            f"Validation dataset 1 has {len(val_dataset_1)} transitions, expected at most {expected_val_episodes_1 * max_steps_per_episode}")
        else:
            self.assertEqual(len(val_dataset_1), 0,
                             "Validation dataset 1 should be empty")

        # Test Case 2: 0% validation split (all data in training)
        val_split_ratio_2 = 0.0
        expected_val_episodes_2 = int(
            num_episodes_total * val_split_ratio_2)  # 0
        expected_train_episodes_2 = num_episodes_total - expected_val_episodes_2  # 10
        print(f"\nTest Case 2: val_split_ratio={val_split_ratio_2}")
        print(
            f"Expected train episodes: {expected_train_episodes_2}, val episodes: {expected_val_episodes_2}")

        train_dataset_2, val_dataset_2 = collect_random_episodes(
            ENV_NAME, num_episodes_total, max_steps_per_episode, image_size, val_split_ratio_2
        )
        self.assertTrue(len(train_dataset_2) >= expected_train_episodes_2 * 1)
        self.assertTrue(len(train_dataset_2) <=
                        expected_train_episodes_2 * max_steps_per_episode)
        self.assertEqual(len(val_dataset_2), 0)

        # Test Case 3: 100% validation split (all data in validation)
        val_split_ratio_3 = 1.0
        expected_val_episodes_3 = int(
            num_episodes_total * val_split_ratio_3)  # 10
        expected_train_episodes_3 = num_episodes_total - expected_val_episodes_3  # 0
        print(f"\nTest Case 3: val_split_ratio={val_split_ratio_3}")
        print(
            f"Expected train episodes: {expected_train_episodes_3}, val episodes: {expected_val_episodes_3}")

        train_dataset_3, val_dataset_3 = collect_random_episodes(
            ENV_NAME, num_episodes_total, max_steps_per_episode, image_size, val_split_ratio_3
        )
        self.assertEqual(len(train_dataset_3), 0)
        self.assertTrue(len(val_dataset_3) >= expected_val_episodes_3 * 1)
        self.assertTrue(len(val_dataset_3) <=
                        expected_val_episodes_3 * max_steps_per_episode)

        # Test Case 4: Small number of episodes
        num_episodes_small = 3
        val_split_ratio_4 = 0.4
        # Implementation logic: split_idx = int((1.0 - val_split_ratio) * total_episodes) for train count
        # train_episodes = total[:split_idx], val_episodes = total[split_idx:]

        split_idx_4 = int((1.0 - val_split_ratio_4) * num_episodes_small)
        expected_train_episodes_4 = split_idx_4
        expected_val_episodes_4 = num_episodes_small - expected_train_episodes_4

        print(
            f"\nTest Case 4: num_episodes={num_episodes_small}, val_split_ratio={val_split_ratio_4}")
        print(f"Calculated split_idx for train: {split_idx_4}")
        print(
            f"Expected train episodes: {expected_train_episodes_4}, val episodes: {expected_val_episodes_4}")

        train_dataset_4, val_dataset_4 = collect_random_episodes(
            ENV_NAME, num_episodes_small, max_steps_per_episode, image_size, val_split_ratio_4
        )

        if expected_train_episodes_4 > 0:
            self.assertTrue(len(train_dataset_4) >=
                            expected_train_episodes_4 * 1)
            self.assertTrue(len(train_dataset_4) <=
                            expected_train_episodes_4 * max_steps_per_episode)
        else:
            self.assertEqual(len(train_dataset_4), 0)

        if expected_val_episodes_4 > 0:
            self.assertTrue(len(val_dataset_4) >= expected_val_episodes_4 * 1)
            self.assertTrue(len(val_dataset_4) <=
                            expected_val_episodes_4 * max_steps_per_episode)
        else:
            self.assertEqual(len(val_dataset_4), 0)

        print("\nFinished test_collect_random_episodes_split.")


if __name__ == '__main__':
    # This allows running the test file directly: python tests/test_data_utils.py
    unittest.main()
