import torch.nn as nn


class Actor_Model(nn.Module):
    def __init__(self, state_dim, action_dim, device=None):
        super(Actor_Model, self).__init__()

        self.device = device if device is not None else 'cpu'

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(-1)
        ).float().to(self.device)

    def forward(self, states):
        return self.nn_layer(states)


class Critic_Model(nn.Module):
    def __init__(self, state_dim, action_dim, device=None):
        super(Critic_Model, self).__init__()
        self.action_dim = action_dim
        self.device = device if device is not None else 'cpu'

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).float().to(self.device)

    def forward(self, states):
        return self.nn_layer(states)
