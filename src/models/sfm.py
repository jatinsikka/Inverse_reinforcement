import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
from utils.buffer import ReplayBuffer, ExpertBuffer
from .networks import SuccessorFeatureNetwork

class SFNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()
        
        # Feature network (phi)
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Successor feature network
        self.sf_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, state, action=None):
        phi = self.feature_net(state)
        
        if action is None:
            return phi
            
        # One-hot encode the action if discrete
        if isinstance(action, int):
            action_onehot = torch.zeros(state.size(0), self.sf_net[0].in_features - state.size(1))
            action_onehot[:, action] = 1
            action = action_onehot
            
        state_action = torch.cat([state, action], dim=1)
        sf = self.sf_net(state_action)
        
        return phi, sf

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        # Ensure state has correct shape
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        return self.policy(state)

class SFM:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Add device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Store config parameters
        self.gamma = config.gamma
        self.tau = config.tau
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        
        # Initialize networks and move to device
        self.policy_net = PolicyNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.sf_net = SuccessorFeatureNetwork(state_dim, action_dim, self.feature_dim, self.hidden_dim).to(self.device)
        self.sf_net_target = SuccessorFeatureNetwork(state_dim, action_dim, self.feature_dim, self.hidden_dim).to(self.device)
        
        # Initialize target network with same weights
        self.sf_net_target.load_state_dict(self.sf_net.state_dict())
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.sf_optimizer = optim.Adam(self.sf_net.parameters(), lr=config.learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size, state_dim, action_dim)
        self.expert_buffer = None

    def select_action(self, state):
        """Select an action given the current state"""
        with torch.no_grad():
            # Handle different input types
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)
            elif isinstance(state, torch.Tensor):
                if state.device != self.device:
                    state = state.to(self.device)
            else:
                raise ValueError(f"Unexpected state type: {type(state)}")
            
            # Ensure state has correct shape [batch_size, state_dim]
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            
            # Get action probabilities
            action_probs = self.policy_net(state)
            action_probs = F.softmax(action_probs, dim=-1)
            
            # Sample action
            dist = Categorical(action_probs)
            action = dist.sample()
            
            return int(action.item())

    def set_expert_buffer(self, expert_buffer):
        self.expert_buffer = expert_buffer

    def update_networks(self, batch_size):
        # Sample from replay buffer
        state, action, next_state, reward, done = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors and move to device
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        # Sample expert transitions and move to device
        expert_state, expert_action = self.expert_buffer.sample(batch_size)
        expert_state = torch.FloatTensor(expert_state).to(self.device)
        expert_action = torch.LongTensor(expert_action).to(self.device)
        
        # Update successor features
        phi, sf = self.sf_net(state, action)
        with torch.no_grad():
            next_phi = self.sf_net_target(next_state)[0]
        
        # Calculate TD error
        target_sf = phi + self.gamma * next_phi * (1 - done.unsqueeze(1))
        sf_loss = F.mse_loss(sf, target_sf.detach())
        
        # Get features for expert and learner states
        expert_phi = self.sf_net(expert_state)[0].detach()
        learner_phi = self.sf_net(state)[0]
        
        # Calculate policy loss (using mean over feature dimension)
        policy_loss = -torch.mean(expert_phi - learner_phi)
        
        # Update networks
        self.sf_optimizer.zero_grad()
        sf_loss.backward()
        self.sf_optimizer.step()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update target network
        for target_param, param in zip(self.sf_net_target.parameters(), self.sf_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        return sf_loss.item(), policy_loss.item()

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'sf_state_dict': self.sf_net.state_dict(),
            'sf_target_state_dict': self.sf_net_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'sf_optimizer_state_dict': self.sf_optimizer.state_dict(),
        }, path)





