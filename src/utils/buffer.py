from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def push(self, state, action, next_state, reward, done):
        """Add a transition to the buffer"""
        # Ensure all inputs are numpy arrays
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        
        # Store the transition
        self.buffer.append((state, action, next_state, reward, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, done = zip(*batch)
        
        # Convert to numpy arrays for efficient transfer to GPU
        return (
            np.array(state),
            np.array(action),
            np.array(next_state),
            np.array(reward, dtype=np.float32),
            np.array(done, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

class ExpertBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.current_episode_states = []
        self.current_episode_actions = []
    
    def add_transition(self, state, action):
        """Add a transition to the current episode"""
        if isinstance(state, np.ndarray):
            state = state.reshape(-1)  # Ensure state is flattened
        self.current_episode_states.append(state)
        self.current_episode_actions.append(action)
    
    def end_trajectory(self):
        """End the current trajectory and add it to the buffer"""
        if self.current_episode_states:
            self.states.extend(self.current_episode_states)
            self.actions.extend(self.current_episode_actions)
            self.current_episode_states = []
            self.current_episode_actions = []
            
            # Trim buffer if it exceeds max size
            if len(self.states) > self.max_size:
                self.states = self.states[-self.max_size:]
                self.actions = self.actions[-self.max_size:]
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        indices = np.random.randint(0, len(self.states), size=batch_size)
        return (
            np.array([self.states[i] for i in indices]),
            np.array([self.actions[i] for i in indices])
        )




