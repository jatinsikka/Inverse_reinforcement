import numpy as np
import torch
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def push(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, next_state, reward, done = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(state)),
            torch.FloatTensor(np.array(action)),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(np.array(reward)),
            torch.FloatTensor(np.array(done))
        )
    
    def __len__(self):
        return len(self.buffer)

class ExpertBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.trajectories = []
        self.current_trajectory = []
        
    def add_transition(self, state, action):
        self.current_trajectory.append((state, action))
    
    def end_trajectory(self):
        if self.current_trajectory:
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = []
            if len(self.trajectories) > self.capacity:
                self.trajectories.pop(0)
    
    def sample_batch(self, batch_size):
        states, actions = [], []
        for _ in range(batch_size):
            traj = random.choice(self.trajectories)
            transition = random.choice(traj)
            states.append(transition[0])
            actions.append(transition[1])
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions))
        )