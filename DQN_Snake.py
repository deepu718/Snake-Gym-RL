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

env = gym.make("SnakeWorld-v0", render_mode=None, size=10)

is_ipython = 'inline' in matplotlib.get_backend()

if is_ipython:
    from Ipython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
torch.manual_seed(seed)
env.reset(seed=seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

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
    
BATCH_SIZE = 512
GAMMA = 0.995
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4


# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100000)


steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []
episode_scores = []

def plot_durations(show_result=False):
    plt.figure(1)
    plt.clf() # Clear the previous frame
    
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    scores_t = torch.tensor(episode_scores, dtype=torch.float)

    # --- TOP GRAPH: DURATION ---
    ax1 = plt.subplot(2, 1, 1) # Create top subplot
    if show_result:
        ax1.set_title('Result')
    else:
        ax1.set_title('Training...')
    
    ax1.set_ylabel('Duration')
    ax1.plot(durations_t.numpy(), color='tab:blue', label='Duration')
    
    # Duration Average (Orange line)
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        ax1.plot(means.numpy(), color='tab:orange', linewidth=2, label='Avg (100 eps)')
    
    ax1.legend(loc='upper left') # Add legend to understand lines
    ax1.grid(True, alpha=0.3)    # Add light grid for readability

    # --- BOTTOM GRAPH: SCORE ---
    ax2 = plt.subplot(2, 1, 2) # Create bottom subplot
    ax2.set_ylabel('Score')
    ax2.set_xlabel('Episode')
    ax2.plot(scores_t.numpy(), color='tab:green', label='Score')

    # Score Average (Red line)
    if len(scores_t) >= 100:
        means_score = scores_t.unfold(0, 100, 1).mean(1).view(-1)
        means_score = torch.cat((torch.zeros(99), means_score))
        ax2.plot(means_score.numpy(), color='tab:red', linewidth=2, label='Avg (100 eps)')

    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # --- THE FIX FOR THE LAYOUT ---
    # hspace=0.4 adds vertical white space between the two graphs
    plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.1)
    
    plt.pause(0.001)
    
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50
best_average_score = -999


for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        print(action)
        
        observation, reward, terminated, truncated, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            current_score = info.get('score', 0) 
            episode_scores.append(current_score)
            if len(episode_scores) >= 100:
                avg_score = sum(episode_scores[-100:]) / 100
                
                # If this is the best average we've seen, save it!
                if avg_score > best_average_score:
                    best_average_score = avg_score
                    torch.save(policy_net.state_dict(), "snake_model_best.pth")
                    print(f"New Best Average Score: {best_average_score:.2f} -> Model Saved!")
            
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

print('Training Complete. Starting video recording of the final run...')

# --- SAVE VIDEO LOGIC ---

# 1. Create a specific environment for recording
# We must use render_mode="rgb_array" for the video wrapper to work
video_env = gym.make("SnakeWorld-v0", render_mode="rgb_array", size=10)

# 2. Wrap the env to record video
# episode_trigger = lambda x: True  -> ensures it records the episode we are about to run
video_env = gym.wrappers.RecordVideo(
    video_env, 
    video_folder="./video_output", 
    name_prefix="snake_final_run",
    episode_trigger=lambda x: True 
)

# 3. Reset and Run one Episode
state, info = video_env.reset(seed=seed)
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

for t in count():
    # --- SELECT ACTION (Greedy / No Randomness) ---
    # We want the best performance for the video, so we disable the epsilon-greedy random check
    with torch.no_grad():
        action = policy_net(state).max(1).indices.view(1, 1)
    
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