import torch
import torch.nn.functional as F
import numpy as np
from .networks import SuccessorFeatureNetwork, PolicyNetwork
from utils.buffer import ReplayBuffer, ExpertBuffer

class SFM:
    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Networks
        self.sf_net = SuccessorFeatureNetwork(
            state_dim, action_dim, config.feature_dim, config.hidden_dim
        ).to(self.device)
        
        self.sf_net_target = SuccessorFeatureNetwork(
            state_dim, action_dim, config.feature_dim, config.hidden_dim
        ).to(self.device)
        self.sf_net_target.load_state_dict(self.sf_net.state_dict())
        
        self.policy = PolicyNetwork(
            state_dim, action_dim, config.hidden_dim
        ).to(self.device)
        
        # Optimizers
        self.sf_optimizer = torch.optim.Adam(self.sf_net.parameters(), lr=config.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # Buffers
        self.replay_buffer = ReplayBuffer(config.buffer_size, state_dim, action_dim)
        self.expert_buffer = ExpertBuffer(1000)
        
        # Initialize reward weights
        self.w = torch.randn(config.feature_dim, requires_grad=True, device=self.device)
        self.w_optimizer = torch.optim.Adam([self.w], lr=config.learning_rate)
    
    def set_expert_buffer(self, expert_buffer):
        self.expert_buffer = expert_buffer

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs = F.softmax(self.policy(state), dim=-1).squeeze(0)  # Remove batch dimension
            if evaluate:
                action = torch.argmax(action_probs)
            else:
                action = torch.multinomial(action_probs, 1).item()  # Sample and convert to scalar
            return action
    
    def update_networks(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample transitions from replay buffer
        state, action, next_state, reward, done = self.replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        # Sample expert transitions
        expert_state, expert_action = self.expert_buffer.sample_batch(batch_size)
        expert_state = expert_state.to(self.device)
        expert_action = expert_action.to(self.device)
        
        # Update successor features
        with torch.no_grad():
            next_phi = self.sf_net_target(next_state)
        
        phi, sf = self.sf_net(state, action)
        target_sf = phi + (1 - done.unsqueeze(-1)) * self.config.gamma * next_phi
        
        sf_loss = F.mse_loss(sf, target_sf.detach())
        
        # Update policy
        expert_phi = self.sf_net(expert_state)[0]
        learner_phi = self.sf_net(state)[0]
        
        policy_loss = -torch.mean(torch.sum(expert_phi - learner_phi, dim=1))
        
        # Update networks
        self.sf_optimizer.zero_grad()
        sf_loss.backward()
        self.sf_optimizer.step()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update target network
        for param, target_param in zip(self.sf_net.parameters(), self.sf_net_target.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
        
        return sf_loss.item(), policy_loss.item()
    
    def save(self, filename):
        torch.save({
            'sf_net': self.sf_net.state_dict(),
            'policy': self.policy.state_dict(),
            'w': self.w
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.sf_net.load_state_dict(checkpoint['sf_net'])
        self.sf_net_target.load_state_dict(checkpoint['sf_net'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.w = checkpoint['w']


