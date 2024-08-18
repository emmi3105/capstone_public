### environment_basic.py
### Define the Job, Candidate, State, and Environment classes for the basic model

import random
import copy
import numpy as np
import pandas as pd
import torch

# Define the Job class 

class Job:
    def __init__(self, job_id, degree_bachelor, degree_master, degree_other, software_programming, 
                 c_programming, python_programming, javascript_programming, professional_software_experience, 
                 management_skills, engineer, minimum_pay):
        self.job_id = job_id
        self.degree_bachelor = degree_bachelor
        self.degree_master = degree_master
        self.degree_other = degree_other
        self.software_programming = software_programming
        self.c_programming = c_programming
        self.python_programming = python_programming
        self.javascript_programming = javascript_programming
        self.professional_software_experience = professional_software_experience
        self.management_skills = management_skills
        self.engineer = engineer
        self.minimum_pay = minimum_pay

        self.allocation_status = 0

# Define the Candidate class

class Candidate:
    def __init__(self, candidate_id, gender, degree_bachelor, degree_master, degree_other, software_programming, 
                 c_programming, python_programming, javascript_programming, professional_software_experience, 
                 management_skills, engineer, previous_pay):
        self.candidate_id = candidate_id
        self.gender = gender
        self.degree_bachelor = degree_bachelor
        self.degree_master = degree_master
        self.degree_other = degree_other
        self.software_programming = software_programming
        self.c_programming = c_programming
        self.python_programming = python_programming
        self.javascript_programming = javascript_programming
        self.professional_software_experience = professional_software_experience
        self.management_skills = management_skills
        self.engineer = engineer
        self.previous_pay = previous_pay

        self.allocation_status = 0

# Define the State class

class State:
    def __init__(self, jobs, candidates, allocation_matrix=None, available_actions=None, num_covariates=12):
        self.candidates = candidates  # a list of candidate objects
        self.jobs = jobs  # a list of job objects
        
        self.num_jobs = len(jobs)
        self.num_candidates = len(candidates)
        self.num_covariates = num_covariates
        
        # The state feature: Matrix with allocation status of jobs and candidates
        self.allocation_matrix = self.get_allocation_matrix() if allocation_matrix is None else allocation_matrix

        # Gender distribution of the candidates that have been allocated to a job
        self.gender_distribution = self.count_gender_alloction()

        # Available actions
        self.available_actions = self.get_available_actions() if available_actions is None else available_actions
    
    def count_gender_alloction(self):
        """Returns a matrix that shows the gender distribution of the candidates that have been allocated to a job."""

        women_count = 0
        men_count = 0

        for i in range(self.num_candidates):
            if self.candidates[i].allocation_status == 1:
                if self.candidates[i].gender == 1:
                    women_count += 1
                elif self.candidates[i].gender == 2:
                    men_count += 1
                else:
                    print("Error")
        
        columns = ["Woman", "Man"]
        df = pd.DataFrame([[women_count, men_count]], columns=columns)

        return df

    def get_allocation_matrix(self):
        """A matrix of shape (num_jobs, num_candidates) with the allocation status of jobs and candidates."""
        allocation_matrix = np.zeros((self.num_jobs, self.num_candidates), dtype=int)

        return allocation_matrix
    

    def get_available_actions(self):
        """A list of tuples, where each tuple is a pair of job and candidate indices."""
        available_actions = []

        for i in range(self.num_jobs):
            for j in range(self.num_candidates):
                # Check if the job and candidate are available
                if self.jobs[i].allocation_status == 0 and self.candidates[j].allocation_status == 0:
                    # Check if the candidate meets the job requirements
                    if (
                        (self.jobs[i].degree_bachelor == 1 and (self.candidates[j].degree_bachelor == 1 or self.candidates[j].degree_master == 1 or self.candidates[j].degree_other == 1)) or
                        (self.jobs[i].degree_master == 1 and (self.candidates[j].degree_master == 1 or self.candidates[j].degree_other == 1)) or
                        (self.jobs[i].degree_other == 1 and self.candidates[j].degree_other == 1) or
                        (self.jobs[i].degree_bachelor == 0 and self.jobs[i].degree_master == 0 and self.jobs[i].degree_other == 0)
                    ) and (
                        self.candidates[j].software_programming >= self.jobs[i].software_programming and
                        self.candidates[j].c_programming >= self.jobs[i].c_programming and
                        self.candidates[j].python_programming >= self.jobs[i].python_programming and
                        self.candidates[j].javascript_programming >= self.jobs[i].javascript_programming and
                        self.candidates[j].professional_software_experience >= self.jobs[i].professional_software_experience and
                        self.candidates[j].management_skills >= self.jobs[i].management_skills and
                        self.candidates[j].engineer >= self.jobs[i].engineer and
                        self.candidates[j].previous_pay <= self.jobs[i].minimum_pay
                    ):
                        # Add the action (i, j) to available actions
                        available_actions.append((i, j))

        return available_actions
    
    def to_tensor(self):
        # Flatten the allocation matrix
        allocation_flat = self.allocation_matrix.flatten()
        state_tensor = torch.tensor(allocation_flat, dtype=torch.float32)
        return state_tensor

    def display_state(self):
        """Display the first ten rows/entries of the three state features."""
        
        print('Allocations:')
        for j, row in enumerate(self.allocation_matrix):
            for i, val in enumerate(row):
                if val == 1:
                    print(f"Job {j} allocated to Candidate {i}")

        print('\nGender distribution:')
        print(self.gender_distribution)

    def display_state_features(self):
        """Display the state feature: allocation_matrix."""

        print(self.allocation_matrix)
 


# Define the Environment class

class Environment:
    def __init__(self, jobs, candidates):
        self.jobs = jobs
        self.candidates = candidates
        self.current_state = State(self.jobs, self.candidates)

        self.num_jobs = self.current_state.num_jobs
        self.num_candidates = self.current_state.num_candidates
        self.num_covariates = self.current_state.num_covariates

        # For normalising the reward
        self.min_reward = -10000
        self.max_reward = 0

        # Initialise the allocation matrix
        self.allocation_matrix = self.current_state.allocation_matrix

        # Initialise the available actions
        self.available_actions = self.current_state.available_actions

        # Initialise the action indices
        self.action_indices = self.initialize_action_indices()

    def initialize_action_indices(self):
        action_indices = {action: idx for idx, action in enumerate(self.available_actions)}
        return action_indices
        
    def reset(self):
        """Reset the environment to its initial state."""
        for job in self.jobs:
            job.allocation_status = 0
        
        for candidate in self.candidates:
            candidate.allocation_status = 0
        
        self.current_state = State(self.jobs, self.candidates)

        # Reset the allocation matrix
        self.allocation_matrix = self.current_state.allocation_matrix

        # Reset the available actions
        self.available_actions = self.current_state.available_actions

        # Reset the action indices
        self.action_indices = self.initialize_action_indices()

        return self.current_state

    def update_available_actions(self, job, candidate):
        """Update available actions by removing the actions involving the job or candidate."""
        self.available_actions = [
            (j, c) for (j, c) in self.available_actions if j != job and c != candidate
        ] 

    def step(self, action):
        done = False

        if len(self.available_actions) != 0:
            job, candidate = action

            # Allocate the job to the candidate
            self.allocation_matrix[job, candidate] = 1
            
            self.jobs[job].allocation_status = 1
            self.candidates[candidate].allocation_status = 1

            # Update available actions by removing the actions involving the job or candidate
            self.update_available_actions(job, candidate)

            # Update the state features
            self.current_state = State(self.jobs, self.candidates, 
                                       allocation_matrix=self.allocation_matrix,
                                       available_actions=self.available_actions)

            if len(self.available_actions) == 0:
                done = True
            
            # Calculate the reward
            reward = self.candidates[candidate].previous_pay - self.jobs[job].minimum_pay
            
            return self.current_state, reward, done