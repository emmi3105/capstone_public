### tools.py
### Define the helper functions for running the experiments

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython import display
from IPython.display import clear_output


# Gender Functions

# How many of the candidates in the sample are men (in percent)
def calculate_sampled_men(sample_candidates):
    male_count = 0

    for candidate in sample_candidates:
        if candidate.gender == 2:
            male_count += 1
    
    return male_count/len(sample_candidates)

# How many of the candidates hired are men (in percent)
def calculate_hired_men(gender_distribution_df):
    men = gender_distribution_df['Man'][0]/gender_distribution_df.iloc[0].sum()
    return men


# DQN Functions

# Get the best schedule
def get_best_allocation(agent, env, show_schedule):
    state = env.reset()
    state_tensor = torch.tensor(state.to_tensor(), dtype=torch.float32).unsqueeze(0)

    done = False
    total_reward = 0

    while not done:
        action = agent.test_select_action(state_tensor, env.current_state.available_actions, env.action_indices)
        
        next_state, reward, done = env.step(action)
        next_state_tensor = torch.tensor(next_state.to_tensor(), dtype=torch.float32).unsqueeze(0)

        total_reward += reward
        state_tensor = next_state_tensor
    
    if show_schedule == True:
        print(next_state.display_state())
        print("Reward: ", total_reward)
    
    # Get the gender_distribution matrix of the final state
    gender_distribution = next_state.gender_distribution

    return gender_distribution, total_reward

# Visualise the loss
def visualize_loss(loss_values):
    #matplotlib.rcdefaults()
    clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Loss Value')
    plt.plot(range(len(loss_values)), loss_values)
    plt.ylim(ymin=0)
    plt.show(block=False)

# Setting a seed for the dqn model
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)