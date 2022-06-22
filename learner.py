import torch

from torch.optim import Adam
from torch.utils.data import DataLoader

from Agents.memory import Memory
from Agents.distribution import Distributions
from Agents.policy_function import PolicyFunction
from Agents.model import Actor_Model, Critic_Model


class Learner():
    def __init__(self, state_dim, action_dim, is_training_mode, policy_kl_range, policy_params, value_clip,
                 entropy_coef, vf_loss_coef, mini_batch, PPO_epochs, gamma, lam, learning_rate, device):
        self.policy_kl_range = policy_kl_range
        self.policy_params = policy_params
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.vf_loss_coef = vf_loss_coef
        self.mini_batch = mini_batch
        self.PPO_epochs = PPO_epochs
        self.is_training_mode = is_training_mode
        self.action_dim = action_dim

        self.actor = Actor_Model(state_dim, action_dim)
        self.actor_old = Actor_Model(state_dim, action_dim)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = Critic_Model(state_dim, action_dim)
        self.critic_old = Critic_Model(state_dim, action_dim)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=learning_rate)

        self.memory = Memory()
        self.policy_function = PolicyFunction(gamma, lam)
        self.distributions = Distributions()

        if is_training_mode:
            self.actor.train()
            self.critic.train()

        else:
            self.actor.eval()
            self.critic.eval()

        self.device = device

    def save_all(self, states, actions, rewards, dones, next_states):
        self.memory.save_all(states, actions, rewards, dones, next_states)

    # Loss for PPO
    def get_loss(self, action_probs, values, old_action_probs, old_values, next_values, actions, rewards, dones):
        # Don't use old value in backpropagation
        Old_values = old_values.detach()

        # Finding the ratio (pi_theta / pi_theta__old):
        logprobs = self.distributions.logprob(action_probs, actions)
        Old_logprobs = self.distributions.logprob(old_action_probs, actions).detach()

        # Getting general advantages estimator
        Advantages = self.policy_function.generalized_advantage_estimation(values, rewards, next_values, dones)
        Returns = (Advantages + values).detach()
        Advantages = ((Advantages - Advantages.mean()) / (Advantages.std() + 1e-6)).detach()

        ratios = (logprobs - Old_logprobs).exp()
        Kl = self.distributions.kl_divergence(old_action_probs, action_probs)

        # Combining TR-PPO with Rollback (Truly PPO)
        pg_loss = torch.where(
            (Kl >= self.policy_kl_range) & (ratios > 1),
            ratios * Advantages - self.policy_params * Kl,
            ratios * Advantages
        )
        pg_loss = pg_loss.mean()

        # Getting entropy from the action probability
        dist_entropy = self.distributions.entropy(action_probs).mean()

        # Getting critic loss by using Clipped critic value
        vpredclipped = Old_values + torch.clamp(values - Old_values, -self.value_clip,
                                                self.value_clip)  # Minimize the difference between old value and new value
        vf_losses1 = (Returns - values).pow(2) * 0.5  # Mean Squared Error
        vf_losses2 = (Returns - vpredclipped).pow(2) * 0.5  # Mean Squared Error
        critic_loss = torch.max(vf_losses1, vf_losses2).mean()

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss
        loss = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss

    # Get loss and Do backpropagation
    def training_ppo(self, states, actions, rewards, dones, next_states):
        action_probs, values = self.actor(states), self.critic(states)
        old_action_probs, old_values = self.actor_old(states), self.critic_old(states)
        next_values = self.critic(next_states)

        loss = self.get_loss(action_probs, values, old_action_probs, old_values, next_values, actions, rewards, dones)

        # === Do backpropagation ===

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # === backpropagation has been finished ===

    # Update the model
    def update_ppo(self):
        batch_size = int(len(self.memory) / self.mini_batch)
        dataloader = DataLoader(self.memory, batch_size, shuffle=False)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):
            for states, actions, rewards, dones, next_states in dataloader:
                self.training_ppo(states.float().to(self.device),
                                  actions.float().to(self.device),
                                  rewards.float().to(self.device),
                                  dones.float().to(self.device),
                                  next_states.float().to(self.device))

        # Clear the memory
        self.memory.clear_memory()

        # Copy new weights into old policy:
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def get_weights(self):
        return self.actor.state_dict()

    def save_weights(self):
        torch.save(self.actor.state_dict(), 'agent.pth')
