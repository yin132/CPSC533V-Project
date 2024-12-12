import gymnasium as gym
import numpy as np

class LunarLander:
    def __init__(self):
        env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5)

    # returns the total reward for running the given policy
    def run(self, policy, debug=False):
        observation, info = self.env.reset()

        total_reward = 0

        episode_over = False
        while not episode_over:
            action = policy(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)

            total_reward += reward

            episode_over = terminated or truncated

        if debug: print(f'total reward: {total_reward}')
        return total_reward