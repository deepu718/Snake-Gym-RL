import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium_env

BATCH_SIZE = 1024
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 10000
TAU = 0.005
LR = 1e-4
NUM_ENVS = 4          # Number of parallel processes
MAX_STEPS = 500000  # Total frames to generate across all envs
BUFFER_SIZE = 1000000

is_ipython = 'inline' in matplotlib.get_backend()

if is_ipython:
    from Ipython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
torch.manual_seed(seed)

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
    
steps_done = 0

def select_action(state,envs):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].cpu().numpy()
    else:
        return envs.action_space.sample()


episode_durations = []
episode_scores = []

def plot_durations(show_result=False):
    plt.figure(1)
    plt.clf() # Clear the previous frame

    if not episode_scores:
        plt.title('Waiting for first episode to finish...')
        plt.pause(0.001)
        return
    
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
    
    ax1.legend(loc='upper left') 
    ax1.grid(True, alpha=0.3)    

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

    
    plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.1)
    
    plt.pause(0.001)
    
    if is_ipython:
        if not show_result:
            display.clear_output(wait=True)
            display.display(plt.gcf())
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

if __name__ == "__main__":
    envs = gym.make_vec(
        "SnakeWorld-v0", 
        num_envs=NUM_ENVS, 
        vectorization_mode="async",
        render_mode = None,
        size = 10 
    )
    envs.reset(seed=seed)
    envs.action_space.seed(seed)
    envs.observation_space.seed(seed)


    n_actions = envs.single_action_space.n
    n_observations = envs.single_observation_space.shape[0]

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(BUFFER_SIZE)

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 10000
    else:
        num_episodes = 50
    
    best_average_score = -999

    current_episode_steps = np.zeros(NUM_ENVS)
    current_episode = 0
    # Force the window to open immediately
    plot_durations() 

    states, infos = envs.reset()
    states = torch.tensor(states, dtype=torch.float32, device=device) # we dont need to unsqueeze here 

    while steps_done < MAX_STEPS:

        actions = select_action(states, envs)

        observations, rewards, terminated, truncated, infos = envs.step(actions)

        dones = terminated | truncated

        current_episode_steps  += 1

        for i in range(NUM_ENVS):
            
            if dones[i]:
                next_state = None

            else:
                next_state = torch.tensor(observations[i], dtype=torch.float32, device=device).unsqueeze(0)
            
            s = states[i].unsqueeze(0)
            a = torch.tensor([[actions[i]]], device=device)
            r = torch.tensor([rewards[i]], device=device)

            memory.push(s,a, next_state, r)

            if dones[i]:
                episode_durations.append(current_episode_steps[i])
                current_episode += 1
                current_episode_steps[i] = 0
                current_score = infos['score'][i]
                episode_scores.append(current_score)
                if len(episode_scores) >= 100:
                    avg_score = sum(episode_scores[-100:]) / 100
                    if avg_score > best_average_score:
                        best_average_score = avg_score
                        torch.save(policy_net.state_dict(), "snake_model_best3.pth")
                        print(f"New Best Average Score: {best_average_score:.2f} -> Model Saved!")

        states = torch.as_tensor(observations, dtype=torch.float32, device=device) # saves memory
        optimize_model()

         # Every 5000 steps, check if Target Network is actually moving
        if steps_done % 5000 == 0:
            # Grab a weight BEFORE update
            before = target_net.layer1.weight.data[0][0].item()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        
        if steps_done % 5000 == 0:
            # Grab weight AFTER update
            after = target_net.layer1.weight.data[0][0].item()
            diff = after - before
            print(f"[Tau Check] Weight: {before:.6f} -> {after:.6f} (Diff: {diff:.8f})")
        
        if current_episode > num_episodes:
            break
        
        if steps_done % 500 == 0:
            plt.pause(0.001)

        # 2. Update the actual graph data (every 1000 steps)
        if steps_done % 1000 == 0:
            print(f"Step {steps_done}: {len(episode_scores)} episodes finished. Plotting...")
            plot_durations()
    
    print('Complete')
    envs.close()
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()
    torch.save(policy_net.state_dict(), "snake_model_best3_1.pth")

