import torch
import gymnasium as gym
from dqn import DQN
from experience_repaly import MemoryReplay
import yaml
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:

    def __init__(self, hyperparameters_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameters_set = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_set[hyperparameters_set]
        
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']





    def run(self, is_training = True, render = False):
        env = gym.make("FrozenLake-v1", render_mode="human" if render else None)

        num_states = env.observation_space.n
        num_actions = env.action_space.n

        reward_for_episode = []
        epsilon_history = []

        policy = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init


        for episode in range(1000):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated:

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy(state).argmax()

                new_state, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.push((state, action, new_state, reward, terminated))
                
                #Move to new state
                state = new_state

            reward_for_episode.append(episode_reward)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)