### agent_basic.py
### Define the RandomAgent, GreedyAgent, and DQNAgent classes

import random
import numpy as np

# DQN Requirements
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple


# Random Agent Class

class RandomAgent:
    def __init__(self):
        pass

    def select_action(self, obs):
        if obs.available_actions:
            action = random.choice(obs.available_actions)
            return action


# Greedy Agent Class

class GreedyAgent:
    def __init__(self):
        pass

    def select_action(self, obs):
        if obs.available_actions:
            max_reward = -np.inf
            best_action = None
            for action in obs.available_actions:
                job, candidate = action
                
                job_payment = obs.jobs[job].minimum_pay
                candidate_payment = obs.candidates[candidate].previous_pay

                reward = candidate_payment - job_payment

                if reward > max_reward:
                    max_reward = reward
                    best_action = action

            return best_action
        

# DQN Agent Classes

# DQN Class: Define the Neural Network architecture
class DQN(nn.Module):
    """Define the Neural Network architecture."""
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128) # input size corresponds to the number of features
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32) 
        self.fc4 = nn.Linear(32, output_size) # output size corresponds to the number of actions
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = F.leaky_relu(self.fc3(x), 0.1)
        x = F.leaky_relu(self.fc4(x), 0.1)
        return x
    
# Replay Buffer Class: The ReplayBuffer stores the transitions that the agent observes
class ReplayBuffer:
    def __init__(self, capacity, env):
        self.capacity = capacity
        self.buffer = [] 
        self.position = 0
        self.env  = env
        
    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer. When we reach the capacity, the position is put to the beginning again."""

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        action_index = self.env.action_indices[action]

        self.buffer[self.position] = Transition(state, action_index, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# DQNAgent Class. The DQNAgent class is the main class that defines the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size, env, batch_size, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        
        self.memory = ReplayBuffer(10000, self.env)

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    
    # ---------------------------------------------------------------------------

    def select_action(self, state, available_actions, action_indices):
        if np.random.rand() <= self.epsilon:
            action_index = np.random.randint(len(available_actions))
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                q_array = q_values.numpy()[0]
                available_q_values = np.array([q_array[action_indices[action]] for action in available_actions])
                action_index = available_q_values.argmax()
            
        return available_actions[action_index]

    def test_select_action(self, state, available_actions, action_indices):
        with torch.no_grad():
            if not available_actions:
                raise ValueError("No available actions to choose from.")
            
            q_values = self.policy_net(state)
            q_array = q_values.numpy()[0]
            available_q_values = np.array([q_array[action_indices[action]] for action in available_actions])
            action_index = available_q_values.argmax()
            
        return available_actions[action_index]  

    # ---------------------------------------------------------------------------

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
    
        state_batch = torch.cat(batch.state).float().to(self.device)
        action_batch = torch.tensor(batch.action).unsqueeze(1).float().to(self.device)
        reward_batch = torch.tensor(batch.reward).float().to(self.device)
        next_state_batch = torch.cat(batch.next_state).float().to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.device)
    
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.long())
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma * (1 - done_batch)) + reward_batch
    
        # Compute loss
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))

        # Zero gradients
        self.optimizer.zero_grad()

        # Perform a backward pass and update the weights
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ---------------------------------------------------------------------------
   
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
