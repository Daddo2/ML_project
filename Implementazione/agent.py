import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import yaml
import random

import matplotlib.pyplot as plt
import pandas as pd 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_one_hot(state, num_states, device=device):
    """state può essere int o tensore scalare; ritorna tensor float su device"""
    if isinstance(state, torch.Tensor):
        state = int(state.detach().cpu().item())
    vec = torch.zeros(num_states, dtype=torch.float32, device=device)
    vec[int(state)] = 1.0
    return vec

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
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.learning_rate = hyperparameters['learning_rate']
        self.discount_factor = hyperparameters['discount_factor']

        self.loss_fn = nn.MSELoss()
        self.optimizer = None


    def run(self, is_training = True, render = False, num_episodes=1000):
        env = gym.make("FrozenLake-v1", render_mode="human" if render else None)

        num_states = env.observation_space.n
        num_actions = env.action_space.n

        reward_for_episode = []
        epsilon_history = []

        policy = DQN(num_states, num_actions).to(device)
        target = DQN(num_states, num_actions).to(device)
        target.load_state_dict(policy.state_dict())
        target.eval()

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.learning_rate)

        step_count = 0

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init


        for episode in range(num_episodes):
            state, _ = env.reset()
            state = to_one_hot(state, num_states, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated:

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy(state.unsqueeze(dim=0)).argmax(dim=1).to(torch.int64)

                new_state, reward, terminated, truncated, info = env.step(action.item())
                episode_reward += reward

                new_state = to_one_hot(new_state, num_states, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.push((state, action, new_state, reward, terminated))
                
                #Move to new state
                state = new_state
                step_count += 1

            reward_for_episode.append(episode_reward)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            if len(memory) > self.mini_batch_size:

                mini_batch = memory.sample(self.mini_batch_size)

                self.train_step(mini_batch, policy, target)

                if step_count > self.network_sync_rate:
                    target.load_state_dict(policy.state_dict())
                    step_count = 0
        
        if is_training:
            torch.save(policy.state_dict(), "frozenlake_dqn_policy.pt")
            print("✅ Modello salvato in 'frozenlake_dqn_policy.pt'")
        
         # Grafico reward per episodio
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(reward_for_episode) + 1), reward_for_episode, label='Reward per episodio')
        plt.xlabel('Episodio')
        plt.ylabel('Reward')
        plt.title('Reward per episodio durante l\'addestramento')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Grafico epsilon decay
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(epsilon_history) + 1), epsilon_history, label='Epsilon')
        plt.xlabel('Episodio')
        plt.ylabel('Epsilon')
        plt.title('Epsilon decay per episodio')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def train_step(self, mini_batch, policy, target):

        states, actions , new_states, rewards, terminations =  zip(*mini_batch)

        states = torch.stack(states).to(device)
        new_states = torch.stack(new_states).to(device)
        actions = torch.tensor(
            [a.item() if isinstance(a, torch.Tensor) else a for a in actions],
            dtype=torch.int64
        ).to(device).view(-1, 1)
        rewards = torch.stack(rewards).float().unsqueeze(1).to(device)
        terminations = torch.tensor(terminations, dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor * target(new_states).max(dim=1)[0]
            
        current_q = policy(states).gather(1, actions)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    agent = Agent("FrozenLake-v1")
    agent.run(is_training=True, render=True)