import torch
from torch.utils.data import Dataset

class ExperienceDataset(Dataset):
    def __init__(self, states, actions, rewards, stop_episodes, sequence_length):
        assert sequence_length is not None and sequence_length > 0


        # states: list of [C, T_i, H, W] tensors; actions, rewards, stop_episodes: length-N lists/vectors
        self.sequence_length = sequence_length

        # concatenate along temporal dim
        self.states = torch.cat(states, dim=1)  # shape [C, total_T, H, W]
        self.states = self.states.squeeze(0)  # remove leading dimension if it exists
        self.actions = torch.stack(actions)     # shape [total_T, ...]
        self.rewards = torch.stack(rewards)     # shape [total_T, ...]
        self.stop_episodes = torch.tensor(stop_episodes, dtype=torch.bool)  # shape [total_T]

        self.valid_start_indices = self.compute_valid_start_indices()

    def compute_valid_start_indices(self):
        N = len(self.stop_episodes)
        valid = []
        # we need [start, start+seq_len) all non‐terminal, and also start+seq_len < N
        for start in range(0, N - self.sequence_length):
            window = self.stop_episodes[start : start + self.sequence_length + 1]
            # all entries in [start, start+seq_len) must be False, and the next one must also exist
            if not window.any():
                valid.append(start)
        return valid

    def __len__(self):
        return len(self.valid_start_indices)

    def __getitem__(self, idx):
        start = self.valid_start_indices[idx]
        end = start + self.sequence_length

        state      = self.states[start:end, :, :]            # [seq_len, H, W]
        next_state = self.states[start+1:end+1, :, :].unsqueeze(0)   # [1, H, W]
        action     = self.actions[start:end]                     # [seq_len, ...]
        reward     = self.rewards[start:end]                     # [seq_len, ...]

        return state, next_state, action, reward
