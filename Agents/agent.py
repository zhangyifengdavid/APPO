import torch

from Agents.memory import Memory
from Agents.model import Actor_Model
from Agents.distribution import Distributions


class Agent:
    def __init__(self, state_dim, action_dim, is_training_mode):
        self.is_training_mode = is_training_mode
        self.device = torch.device('cpu')

        self.memory = Memory()
        self.distributions = Distributions(self.device)
        self.actor = Actor_Model(state_dim, action_dim, self.device)

        if is_training_mode:
            self.actor.train()
        else:
            self.actor.eval()

    def save_eps(self, state, action, reward, done, next_state):
        self.memory.save_eps(state, action, reward, done, next_state)

    def get_all(self):
        return self.memory.get_all()

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device).detach()
        action_probs = self.actor(state)

        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action = self.distributions.sample(action_probs)
        else:
            action = torch.argmax(action_probs, 1)

        return action.cpu().item()

    def set_weights(self, weights):
        self.actor.load_state_dict(weights)

    def load_weights(self):
        self.actor.load_state_dict(torch.load('agent.pth', map_location=self.device))