import ray
import gym

from Agents.agent import Agent

@ray.remote
class Runner():
    def __init__(self, env_name, training_mode, render, n_update, tag):
        self.env = gym.make(env_name)
        self.states = self.env.reset()

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.agent = Agent(self.state_dim, self.action_dim, training_mode)

        self.render = render
        self.tag = tag
        self.training_mode = training_mode
        self.n_update = n_update

    def run_episode(self, i_episode, total_reward, eps_time):
        self.agent.load_weights()

        for _ in range(self.n_update):
            action = int(self.agent.act(self.states))
            next_state, reward, done, _ = self.env.step(action)

            eps_time += 1
            total_reward += reward

            if self.training_mode:
                self.agent.save_eps(self.states.tolist(), action, reward, float(done), next_state.tolist())

            self.states = next_state

            if self.render:
                self.env.render()

            if done:
                self.states = self.env.reset()
                i_episode += 1
                print('Episode {} \t t_reward: {} \t time: {} \t process no: {} \t'.format(i_episode, total_reward,
                                                                                           eps_time, self.tag))

                total_reward = 0
                eps_time = 0

        return self.agent.get_all(), i_episode, total_reward, eps_time, self.tag