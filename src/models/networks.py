import torch
import torch.nn as nn

class SuccessorFeatureNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.sf_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, state, action=None):
        phi = self.feature_net(state)
        if action is None:
            return phi
        
        state_action = torch.cat([state, action], dim=-1)
        sf = self.sf_net(state_action)
        return phi, sf

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.net(state)  # Returns logits of shape [batch_size, action_dim]
