import numpy as np

from torch.utils.data import Dataset


class Memory(Dataset):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype=np.float32), np.array(self.actions[idx], dtype=np.float32), np.array(
            [self.rewards[idx]], dtype=np.float32), \
               np.array([self.dones[idx]], dtype=np.float32), np.array(self.next_states[idx], dtype=np.float32)

    def get_all(self):
        return self.states, self.actions, self.rewards, self.dones, self.next_states

    def save_all(self, states, actions, rewards, dones, next_states):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.next_states = next_states

    def save_eps(self, state, action, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]