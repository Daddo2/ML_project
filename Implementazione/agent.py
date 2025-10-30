import torch
import gymnasium as gym
from dqn import DQN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def run(self, is_training = True, render = False):
        env = gym.make("FrozenLake-v1", render_mode="human" if render else None)

        num_states = env.observation_space.n
        num_actions = env.action_space.n

        policy = DQN(num_states, num_actions).to(device)

        state, _ = env.reset()
        while True:

            action = env.action_space.sample()

            state, reward, terminated, truncated, info = env.step(action)

            if terminated:
                break

        env.close()