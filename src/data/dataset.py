import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class ExperienceDataset(Dataset):
    def __init__(self, states, actions, rewards, stop_episodes, sequence_length):
        assert sequence_length is not None and sequence_length > 0

        # states: list of [C, T_i, H, W] tensors; actions, rewards, stop_episodes: length-N lists/vectors
        self.sequence_length = sequence_length

        # concatenate along temporal dim → [C, total_T, H, W]
        self.states = torch.cat(states, dim=1)
        self.actions = torch.stack(actions)     # [total_T, …]
        self.rewards = torch.stack(rewards)     # [total_T, …]
        self.stop_episodes = torch.tensor(stop_episodes, dtype=torch.bool)  # [total_T]

        # 1) resize spatial dims to 224×224
        #    our tensor is [C, total_T, H, W], so we treat total_T as extra channels:
        C, T, H, W = self.states.shape
        # permute to [T, C, H, W] → interpolate → back to [C, T, 224, 224]
        st = self.states.permute(1, 0, 2, 3)                   # [T, C, H, W]
        st = F.interpolate(st, size=(224, 224),
                           mode='bilinear', align_corners=False)
        st = st.permute(1, 0, 2, 3)                            # [C, T, 224, 224]
        self.states = st

        # 2) ensure 3‑channel input for ResNet
        C, T, H, W = self.states.shape
        if C == 1:
            # duplicate the single channel 3 times
            self.states = self.states.repeat(3, 1, 1, 1)
        elif C != 3:
            raise ValueError(f"Expected 1 or 3 input channels, got {C}")

        # precompute valid sequence start indices
        self.valid_start_indices = self.compute_valid_start_indices()

    def compute_valid_start_indices(self):
        N = len(self.stop_episodes)
        valid = []
        # want [start, start+seq_len) all non‐terminal
        for start in range(0, N - self.sequence_length):
            # look at flags from start up to and including start+seq_len
            window = self.stop_episodes[start : start + self.sequence_length + 1]
            if not window.any():
                valid.append(start)
        return valid

    def __len__(self):
        return len(self.valid_start_indices)

    def __getitem__(self, idx):
        start = self.valid_start_indices[idx]
        end = start + self.sequence_length

        # state sequence: [C, seq_len, 224, 224]
        state      = self.states[:, start:end, :, :]
        # next frame: [C, 1, 224, 224]
        next_state = self.states[:, end:end+1, :, :]

        action     = self.actions[start:end]   # [seq_len, …]
        reward     = self.rewards[start:end]   # [seq_len, …]

        return state, next_state, action, reward
