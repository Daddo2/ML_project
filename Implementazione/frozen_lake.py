import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import Wrapper
from tqdm import trange
import imageio
import pygame

class CustomFrozenLakeWrapper(Wrapper):
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        if terminated and reward == 1.0:
            reward = 100.0
        elif terminated and reward == 0.0:
            reward = -100.0
        else:
            reward = -1.0
        return next_state, reward, terminated, truncated, info

# Config
ENV_ID = "FrozenLake-v1"
MAP_NAME = "4x4"        # "4x4" o "8x8"
IS_SLIPPERY = False
SEED = 0

# Create results dir
SAVE_DIR = "results_qlearning"
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize env
env = CustomFrozenLakeWrapper(gym.make(ENV_ID, map_name=MAP_NAME, is_slippery=IS_SLIPPERY, render_mode="human"))
env.reset(seed=SEED)

n_states = env.observation_space.n
n_actions = env.action_space.n

print(f"Environment: {ENV_ID}, map: {MAP_NAME}, slippery: {IS_SLIPPERY}")
print(f"States: {n_states}, Actions: {n_actions}")


#class CustomFrozenLakeWrapper(Wrapper):
    #def step(self, action):
        #next_state, reward, terminated, truncated, info = self.env.step(action)
        #if terminated and reward == 1.0:
            #reward = 100.0
        #elif terminated and reward == 0.0:
            #reward = -100.0
        #else:
            #reward = -1.0
        #return next_state, reward, terminated, truncated, info

# Step 3: Q-table e policy
Q = np.zeros((n_states, n_actions), dtype=float)

# Epsilon-greedy
def choose_action(state, eps):
    if np.random.random() < eps:
        return env.action_space.sample()   # esplora
    else:
        return int(np.argmax(Q[state]))    # sfrutta (azione con Q massima)


# ============================
# STEP 4: Aggiornamento Q (regola di Bellman)
# ============================
ALPHA = 0.8   # learning rate
GAMMA = 0.95  # discount factor

def update_q(state, action, reward, next_state, done):
    best_next = np.max(Q[next_state])
    target = reward + (0.0 if done else GAMMA * best_next)
    Q[state, action] = (1 - ALPHA) * Q[state, action] + ALPHA * target


# ============================
# STEP 5: Ciclo di training principale
# ============================
NUM_EPISODES = 5000
MAX_STEPS_PER_EP = 100
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_RATE = 0.999  # moltiplicativo per episodio

eps = EPS_START
rewards_all = []
successes = []

print("Inizio training...")

for ep in trange(1, NUM_EPISODES + 1):
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < MAX_STEPS_PER_EP:
        action = choose_action(state, eps)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        update_q(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        steps += 1

    # decay epsilon
    eps = max(EPS_END, eps * EPS_DECAY_RATE)

    rewards_all.append(total_reward)
    successes.append(1 if total_reward > 0 else 0)

print(f"Reward totale (somma su tutti gli episodi): {np.sum(rewards_all)}")
print(f"Reward medio per episodio: {np.mean(rewards_all):.4f}")

# Salvataggio Q-table
np.save(os.path.join(SAVE_DIR, "q_table.npy"), Q)
print("Training finito. Q-table salvata in:", os.path.join(SAVE_DIR, "q_table.npy"))

# ============================
# STEP 6: Grafici performance
# ============================
window = 100
if len(rewards_all) >= window:
    mean_reward_window = np.convolve(rewards_all, np.ones(window)/window, mode='valid')
    x = np.arange(window, window + len(mean_reward_window))
else:
    mean_reward_window = rewards_all
    x = np.arange(1, 1 + len(mean_reward_window))

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(x, mean_reward_window)
plt.title(f"Moving average reward (window={window})")
plt.xlabel("Episode")
plt.ylabel("Avg reward")

cum_success = np.cumsum(successes) / np.arange(1, len(successes)+1)
plt.subplot(1,2,2)
plt.plot(cum_success)
plt.title("Cumulative success rate")
plt.xlabel("Episode")
plt.ylabel("Success rate")

plt.tight_layout()
plot_path = os.path.join(SAVE_DIR, "performance.png")
plt.savefig(plot_path)
plt.show()
print("Grafico salvato in:", plot_path)

# ============================
# STEP 7: Creazione GIF policy finale
# ============================
def run_episode_and_capture(max_steps=100):
    """Esegue un episodio greedy e cattura i frame per creare una GIF"""
    frames = []
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    while not done and steps < max_steps:
        action = int(np.argmax(Q[state]))  # greedy
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        state = next_state
        frames.append(env.render())
    return frames, total_reward, steps

frames, r, s = run_episode_and_capture()
if frames:
    gif_path = os.path.join(SAVE_DIR, "policy_greedy.gif")
    imageio.mimsave(gif_path, frames, fps=4)
    print(f"GIF salvata: {gif_path} (reward={r}, steps={s})")
else:
    print("Nessun frame catturato (controlla render_mode=rgb_array).")

env.close()
print("Esecuzione completata.")

