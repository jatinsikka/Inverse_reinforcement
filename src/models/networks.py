import torch
import torch.nn as nn

class SuccessorFeatureNetwork(nn.Module):
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
        
        self.action_dim = action_dim
    
    def forward(self, state, action=None):
        phi = self.feature_net(state)
        
        if action is None:
            return phi
            
        # Convert action to one-hot encoding if it's not already
        if len(action.shape) == 1:  # If action is 1D tensor of indices
            action_onehot = torch.zeros(action.size(0), self.action_dim, device=state.device)
            action_onehot.scatter_(1, action.unsqueeze(1), 1)
        else:
            action_onehot = action
            
        state_action = torch.cat([state, action_onehot], dim=1)
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

