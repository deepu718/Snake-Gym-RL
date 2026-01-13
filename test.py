import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium_env # This imports your folder and runs the register code
# from DQN_Snake import DQN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.eval()
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128,128)
        self.layer3 = nn.Linear(128,128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
video_env = gym.make("SnakeWorld-v0", render_mode="rgb_array", size=10)

n_observations = video_env.observation_space.shape[0]
n_actions = video_env.action_space.n

model = DQN(n_observations, n_actions).to(device)
model.load_state_dict(torch.load("snake_model_best3.pth", map_location=device))
model.eval()

# 2. Wrap the env to record video
# episode_trigger = lambda x: True  -> ensures it records the episode we are about to run
video_env = gym.wrappers.RecordVideo(
    video_env, 
    video_folder="./video_output", 
    name_prefix="snake_final_run2",
    episode_trigger=lambda x: True 
)

# 3. Reset and Run one Episode
state, info = video_env.reset(seed=42)
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

for t in count():
    # --- SELECT ACTION (Greedy / No Randomness) ---
    # We want the best performance for the video, so we disable the epsilon-greedy random check
    with torch.no_grad():
        action = model(state).max(1).indices.view(1, 1)
    
    # --- STEP ---
    observation, reward, terminated, truncated, _ = video_env.step(action.item())
    
    # Check if done
    done = terminated or truncated
    
    if not done:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        state = next_state
    else:
        break

video_env.close()
print("Video saved in the './video_output' folder!")

