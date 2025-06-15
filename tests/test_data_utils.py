import unittest
import sys # For mocking modules
from unittest.mock import patch, MagicMock, call
import numpy as np

# Mock stable_baselines3 and related modules BEFORE they are imported by src.utils.data_utils
# This is to avoid ModuleNotFoundError if stable_baselines3 is not installed in test env.
MOCK_SB3 = MagicMock()
sys.modules['stable_baselines3'] = MOCK_SB3
sys.modules['stable_baselines3.common.vec_env'] = MagicMock()
sys.modules['stable_baselines3.common.callbacks'] = MagicMock()
sys.modules['stable_baselines3.common.logger'] = MagicMock()
sys.modules['stable_baselines3.common.monitor'] = MagicMock()
sys.modules['stable_baselines3.common.policies'] = MagicMock()
sys.modules['stable_baselines3.common.save_util'] = MagicMock()


# Now import the modules to be tested
import torch
import gymnasium as gym # For action_space.sample type hinting if needed & env spec
from torch.utils.data import Dataset # For type hinting

# Functions to test
from src.utils.data_utils import collect_random_episodes, ExperienceDataset
from src.utils import data_utils # To allow patching attributes like pickle directly in data_utils

# Minimal config for tests
def get_dummy_config(load_path="", save_name="test_dataset.pkl", num_episodes=1):
    return {
        'environment_name': 'TestEnv-v0',
        'num_episodes_data_collection': num_episodes,
        'dataset_dir': 'test_datasets/', # Should be a temporary directory for tests
        'load_dataset_path': load_path,
        'dataset_filename': save_name,
        'image_size': 32
    }

class TestFrameSkipping(unittest.TestCase):

    def _create_mock_env(self, initial_state_data, step_return_data_sequence, action_sequence=None):
        """
        Helper to create a mock environment.
        initial_state_data: Data returned by env.reset() (e.g., np.array([0]))
        step_return_data_sequence: A list of tuples, each for one env.step() call:
                                   (next_state_img, reward, terminated, truncated, info)
        action_sequence: A list of actions to be returned by env.action_space.sample()
        """
        mock_env = MagicMock(spec=gym.Env)
        mock_env.render_mode = 'rgb_array' # To satisfy checks in data_utils
        mock_env.reset.return_value = (initial_state_data, {}) # (obs, info)

        # Ensure step returns are individual tuples, not a list of one tuple
        if len(step_return_data_sequence) == 1 and isinstance(step_return_data_sequence[0], list):
             mock_env.step.side_effect = step_return_data_sequence[0]
        else:
            mock_env.step.side_effect = step_return_data_sequence

        mock_env.action_space = MagicMock(spec=gym.spaces.Discrete) # Assuming discrete for sample
        if action_sequence:
            mock_env.action_space.sample.side_effect = action_sequence
        else:
            mock_env.action_space.sample.return_value = 0 # Default action if not specified

        # For environments that might have a spec attribute
        mock_env.spec = MagicMock()
        mock_env.spec.max_episode_steps = None # Or some large number

        return mock_env

    @patch('src.utils.data_utils.os.makedirs') # Mock makedirs to prevent actual directory creation
    @patch('src.utils.data_utils.os.path.exists')
    @patch('src.utils.data_utils.pickle.dump')
    @patch('src.utils.data_utils.gym.make')
    def test_no_frame_skipping(self, mock_gym_make, mock_pickle_dump, mock_os_path_exists, mock_os_makedirs):
        """Test data collection with frame_skipping = 0."""
        mock_os_path_exists.return_value = False # Force new data collection
        mock_os_makedirs.return_value = None # Mock directory creation

        initial_state = np.array([[[0]]], dtype=np.uint8) # HWC format (1x1x1 image)

        # Sequence of (next_state, reward, terminated, truncated, info)
        episode_step_data = [
            (np.array([[[1]]], dtype=np.uint8), 1.0, False, False, {}), # s0 -> s1 (action a0)
            (np.array([[[2]]], dtype=np.uint8), 2.0, False, False, {}), # s1 -> s2 (action a1)
            (np.array([[[3]]], dtype=np.uint8), 3.0, True, False, {}),   # s2 -> s3 (action a2), terminated
        ]
        actions = [10, 11, 12] # a0, a1, a2

        mock_env = self._create_mock_env(initial_state, episode_step_data, actions)
        mock_gym_make.return_value = mock_env

        config = get_dummy_config(num_episodes=1) # Collect 1 episode
        max_steps_per_episode = 10

        train_dataset, val_dataset = collect_random_episodes(
            config,
            max_steps_per_episode=max_steps_per_episode,
            image_size=(1,1), # Keep image size minimal
            validation_split_ratio=0.0, # All data to train_dataset
            frame_skipping=0
        )

        self.assertEqual(len(train_dataset), 3, "Should have 3 transitions for 3 steps.")
        self.assertEqual(len(val_dataset), 0)

        # Expected transitions:
        # (s0, a0, r1, s1)
        # (s1, a1, r2, s2)
        # (s2, a2, r3, s3)

        # Raw states before transforms
        expected_raw_states = [initial_state, episode_step_data[0][0], episode_step_data[1][0]]
        expected_raw_next_states = [episode_step_data[0][0], episode_step_data[1][0], episode_step_data[2][0]]
        expected_rewards = [step[1] for step in episode_step_data] # 1.0, 2.0, 3.0

        for i in range(len(train_dataset)):
            s, a, r, ns = train_dataset[i]

            # Note: ExperienceDataset applies T.ToPILImage(), T.Resize(), T.ToTensor().
            # For a 1x1x1 uint8 array, ToTensor changes it to 1x1x1 float tensor.
            # Values should be scaled (e.g. /255.0), but for simple integer states, they might be direct.
            # Let's verify the core values.
            self.assertTrue(torch.is_tensor(s), "State should be a tensor")
            self.assertTrue(torch.is_tensor(ns), "Next state should be a tensor")

            # Check the value. Assuming ToTensor scales 0-255 to 0-1.
            # For simple integer data like np.array([[[0]]]), after ToPILImage -> Resize -> ToTensor,
            # the value might become itself if it's already small, or scaled.
            # Account for ToTensor scaling uint8 0-255 to float32 0-1
            self.assertAlmostEqual(s.numpy()[0,0,0], expected_raw_states[i][0,0,0] / 255.0, places=5, msg=f"State mismatch at index {i}")
            self.assertEqual(a.item(), actions[i], f"Action mismatch at index {i}")
            self.assertEqual(r.item(), expected_rewards[i], f"Reward mismatch at index {i}")
            self.assertAlmostEqual(ns.numpy()[0,0,0], expected_raw_next_states[i][0,0,0] / 255.0, places=5, msg=f"Next state mismatch at index {i}")

        # Verify pickle.dump was called (or not, depending on test goals for saving)
        # For this test, primarily focused on collection logic.
        # mock_pickle_dump.assert_called_once() # If you expect saving

    # Test Case 2: Frame Skipping (frame_skipping = 1)
    @patch('src.utils.data_utils.os.makedirs')
    @patch('src.utils.data_utils.os.path.exists')
    @patch('src.utils.data_utils.pickle.dump')
    @patch('src.utils.data_utils.gym.make')
    def test_with_frame_skipping_one(self, mock_gym_make, mock_pickle_dump, mock_os_path_exists, mock_os_makedirs):
        mock_os_path_exists.return_value = False
        mock_os_makedirs.return_value = None

        initial_state = np.array([[[0]]], dtype=np.uint8) # s0

        # s0 --(a0, r1)--> s1 --(a_skip1, r2)--> s2 (end of 1st recorded step)
        # s2 --(a1, r3)--> s3 --(a_skip2, r4)--> s4 (end of 2nd recorded step)
        # s4 --(a2, r5)--> s5 (terminated, part of 3rd recorded step, no skip)
        episode_step_data = [
            (np.array([[[1]]], dtype=np.uint8), 1.0, False, False, {}), # s0 -> s1 (action a0, reward r1)
            (np.array([[[2]]], dtype=np.uint8), 2.0, False, False, {}), # s1 -> s2 (action a_skip1, reward r2)
            (np.array([[[3]]], dtype=np.uint8), 3.0, False, False, {}), # s2 -> s3 (action a1, reward r3)
            (np.array([[[4]]], dtype=np.uint8), 4.0, False, False, {}), # s3 -> s4 (action a_skip2, reward r4)
            (np.array([[[5]]], dtype=np.uint8), 5.0, True,  False, {}), # s4 -> s5 (action a2, reward r5) - terminates
        ]
        # Actions taken: a0 (for s0), a_skip1 (for s1), a1 (for s2), a_skip2 (for s3), a2 (for s4)
        actions_env_samples = [10, 100, 11, 110, 12]

        mock_env = self._create_mock_env(initial_state, episode_step_data, actions_env_samples)
        mock_gym_make.return_value = mock_env

        config = get_dummy_config(num_episodes=1)
        max_steps_per_episode = 20

        train_dataset, _ = collect_random_episodes(
            config, max_steps_per_episode, image_size=(1,1),
            validation_split_ratio=0.0, frame_skipping=1
        )

        # Expected transitions with frame_skipping=1:
        # 1. (s0, a0, r1+r2, s2)
        # 2. (s2, a1, r3+r4, s4)
        # 3. (s4, a2, r5,    s5) - no skip as episode ends
        self.assertEqual(len(train_dataset), 3)

        expected_recorded_states_raw = [initial_state, episode_step_data[1][0], episode_step_data[3][0]] # s0, s2, s4
        expected_recorded_actions = [actions_env_samples[0], actions_env_samples[2], actions_env_samples[4]] # a0, a1, a2
        expected_accumulated_rewards = [
            episode_step_data[0][1] + episode_step_data[1][1], # r1+r2
            episode_step_data[2][1] + episode_step_data[3][1], # r3+r4
            episode_step_data[4][1]                            # r5
        ]
        expected_recorded_next_states_raw = [episode_step_data[1][0], episode_step_data[3][0], episode_step_data[4][0]] # s2, s4, s5

        for i in range(len(train_dataset)):
            s, a, r, ns = train_dataset[i]
            self.assertAlmostEqual(s.numpy()[0,0,0], expected_recorded_states_raw[i][0,0,0] / 255.0, places=5, msg=f"State mismatch at index {i}")
            self.assertEqual(a.item(), expected_recorded_actions[i], f"Action mismatch at index {i}")
            self.assertAlmostEqual(r.item(), expected_accumulated_rewards[i], places=5, msg=f"Reward mismatch at index {i}")
            self.assertAlmostEqual(ns.numpy()[0,0,0], expected_recorded_next_states_raw[i][0,0,0] / 255.0, places=5, msg=f"Next state mismatch at index {i}")

    # Test Case 3: Frame Skipping with Episode End during skip
    @patch('src.utils.data_utils.os.makedirs')
    @patch('src.utils.data_utils.os.path.exists')
    @patch('src.utils.data_utils.pickle.dump')
    @patch('src.utils.data_utils.gym.make')
    def test_frame_skipping_ends_mid_skip(self, mock_gym_make, mock_pickle_dump, mock_os_path_exists, mock_os_makedirs):
        mock_os_path_exists.return_value = False
        mock_os_makedirs.return_value = None

        initial_state = np.array([[[10]]], dtype=np.uint8) # s10

        # s10 --(a0, r1)--> s11
        # s11 --(a_skip1, r2)--> s12 (this step terminates during a skip for frame_skipping=2)
        episode_step_data = [
            (np.array([[[11]]], dtype=np.uint8), 1.0, False, False, {}), # s10 -> s11 (action a0, reward r1)
            (np.array([[[12]]], dtype=np.uint8), 2.0, True,  False, {}), # s11 -> s12 (action a_skip1, reward r2) - terminates
        ]
        actions_env_samples = [20, 200]

        mock_env = self._create_mock_env(initial_state, episode_step_data, actions_env_samples)
        mock_gym_make.return_value = mock_env

        config = get_dummy_config(num_episodes=1)
        max_steps_per_episode = 20

        train_dataset, _ = collect_random_episodes(
            config, max_steps_per_episode, image_size=(1,1),
            validation_split_ratio=0.0, frame_skipping=2 # Try to skip 2 frames
        )

        # Expected behavior:
        # Main step: (s10, a0, r1, s11)
        # Skip 1: (s11, a_skip1, r2, s12) -> Terminates.
        # Frame skipping loop should break.
        # Recorded transition: (s10, a0, r1+r2, s12)
        self.assertEqual(len(train_dataset), 1)

        s, a, r, ns = train_dataset[0]
        self.assertAlmostEqual(s.numpy()[0,0,0], initial_state[0,0,0] / 255.0, places=5)
        self.assertEqual(a.item(), actions_env_samples[0]) # Original action a0
        self.assertAlmostEqual(r.item(), episode_step_data[0][1] + episode_step_data[1][1], places=5) # r1+r2
        self.assertAlmostEqual(ns.numpy()[0,0,0], episode_step_data[1][0][0,0,0] / 255.0, places=5) # s12

    # Test Case 4: Metadata Check
    # Patching pickle.dump within the data_utils module where it's called
    # REMOVED @patch('src.utils.data_utils.os.makedirs') to allow directory creation
    @patch('src.utils.data_utils.os.path.exists')
    @patch('src.utils.data_utils.pickle.dump') # Correct path for pickle.dump
    @patch('src.utils.data_utils.gym.make')
    def test_frame_skipping_metadata_saved(self, mock_gym_make, mock_pickle_dump, mock_os_path_exists): # mock_os_makedirs removed
        mock_os_path_exists.return_value = False # Force collection and save
        # mock_os_makedirs is removed to allow actual directory creation if needed by pickle.dump's path.

        initial_state = np.array([[[0]]], dtype=np.uint8)
        episode_step_data = [
            (np.array([[[1]]], dtype=np.uint8), 1.0, True, False, {}), # Single step episode
        ]
        actions = [50]

        mock_env = self._create_mock_env(initial_state, episode_step_data, actions)
        mock_gym_make.return_value = mock_env

        test_frame_skipping_value = 3
        config = get_dummy_config(save_name="metadata_test.pkl")

        collect_random_episodes(
            config, max_steps_per_episode=5, image_size=(1,1),
            validation_split_ratio=0.0, frame_skipping=test_frame_skipping_value
        )

        mock_pickle_dump.assert_called_once()
        args, kwargs = mock_pickle_dump.call_args
        saved_data = args[0] # The object that was dumped

        self.assertIn('metadata', saved_data)
        metadata = saved_data['metadata']
        self.assertIn('frame_skipping', metadata)
        self.assertEqual(metadata['frame_skipping'], test_frame_skipping_value)

if __name__ == '__main__':
    unittest.main()
