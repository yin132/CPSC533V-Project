import gymnasium as gym
import numpy as np

class Cartpole:
    def __init__(self, num_bins):
        self.env = gym.make("CartPole-v1")

        # Want to keep effective range as (-2.4, 2.4) for position and 
        # (-0.2095, 0.2095) for angle since it ends outside that range
        # same as observation space
        self.min_observation = np.array([-2.4, -2, -0.2095, -2])
        self.max_observation = np.array([2.4, 2, 0.2095, 2])

        self.num_bins = num_bins

    # returns the total reward for running the given policy
    def run(self, policy, debug=False):
        observation, info = self.env.reset()

        total_reward = 0

        episode_over = False
        while not episode_over:
            # Tabular
            if self.num_bins != 0:
                observation = self.normalize_observation(observation)
                observation = self.bin_observation(observation)

            action = policy(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)

            total_reward += reward

            episode_over = terminated or truncated

        if debug: print(f'total reward: {total_reward}')
        return total_reward

    # normalizes observation to [0, 1] for each element
    def normalize_observation(self, observation):
        # clip the bounds of the observation
        observation = np.maximum(observation, self.min_observation)
        observation = np.minimum(observation, self.max_observation)

        # scale the observation
        observation = (self.max_observation - observation) / (self.max_observation - self.min_observation)

        return observation

    # returns which bin each of the elements is in in range [0, num_bins)
    def bin_observation(self, normalized_observation):
        binned_observation = np.floor(normalized_observation * self.num_bins)

        # in case of being at max need to subtract 1 from those at num_bins
        binned_observation[binned_observation == self.num_bins] -= 1

        return binned_observation