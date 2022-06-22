import gc
import gym
import ray
import time
import datetime

from runner import Runner
from learner import Learner


def main():
    """
    Hyper-parameters
    """
    #############################################
    training_mode = True  # If you want to train the agent, set this to True. But set this otherwise if you only want to test it

    render = True  # If you want to display the image, set this to True. Turn this off if you run this in Google Collab
    n_update = 128  # How many episode before you update the Policy. Recommended set to 1024 for Continuous
    n_episode = 1000  # How many episode you want to run
    n_agent = 2  # How many agent you want to run asynchronously

    policy_kl_range = 0.0008  # Recommended set to 0.03 for Continuous
    policy_params = 20  # Recommended set to 5 for Continuous
    value_clip = 1.0  # How many value will be clipped. Recommended set to the highest or lowest possible reward
    entropy_coef = 0.05  # How much randomness of action you will get. Because we use Standard Deviation for Continous, no need to use Entropy for randomness
    vf_loss_coef = 1.0  # Just set to 1
    mini_batch = 4  # How many batch per update. size of batch = n_update / minibatch. Recommended set to 32 for Continous
    PPO_epochs = 4  # How many epoch per update. Recommended set to 10 for Continuous

    gamma = 0.99  # Just set to 0.99
    lam = 0.95  # Just set to 0.95
    learning_rate = 2.5e-4  # Just set to 0.95
    #############################################

    device = 'cpu'
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    learner = Learner(state_dim, action_dim, training_mode, policy_kl_range, policy_params,
                      value_clip, entropy_coef, vf_loss_coef, mini_batch, PPO_epochs,
                      gamma, lam, learning_rate, device)
    #############################################
    start = time.time()
    ray.init()
    try:
        runners = [Runner.remote(env_name, training_mode, render, n_update, i) for i in range(n_agent)]
        learner.save_weights()

        episode_ids = []
        for i, runner in enumerate(runners):
            episode_ids.append(runner.run_episode.remote(i, 0, 0))
            time.sleep(1)

        for _ in range(1, n_episode + 1):
            ready, not_ready = ray.wait(episode_ids)
            trajectory, i_episode, total_reward, eps_time, tag = ray.get(ready)[0]

            states, actions, rewards, dones, next_states = trajectory
            learner.save_all(states, actions, rewards, dones, next_states)

            learner.update_ppo()
            learner.save_weights()

            episode_ids = not_ready
            episode_ids.append(runners[tag].run_episode.remote(i_episode, total_reward, eps_time))

            gc.collect()

    except KeyboardInterrupt:
        print('\nTraining has been Shutdown \n')

    finally:
        ray.shutdown()

        finish = time.time()
        timedelta = finish - start
        print('Time length: {}'.format(str(datetime.timedelta(seconds=timedelta))))


if __name__ == '__main__':
    main()
