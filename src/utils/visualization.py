import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

class Visualizer:
    def __init__(self):
        self.rewards = []
        self.losses = []
        
    def update(self, episode_reward, sf_loss, policy_loss):
        self.rewards.append(episode_reward)
        self.losses.append((sf_loss, policy_loss))
    
    def plot(self):
        clear_output(True)
        plt.figure(figsize=(20, 5))
        
        # Plot rewards
        plt.subplot(131)
        plt.title('Episode Rewards')
        plt.plot(self.rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot losses
        plt.subplot(132)
        sf_losses = [x[0] for x in self.losses]
        policy_losses = [x[1] for x in self.losses]
        
        plt.title('Training Losses')
        plt.plot(sf_losses, label='SF Loss')
        plt.plot(policy_losses, label='Policy Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_plots(self, filename):
        plt.figure(figsize=(20, 5))
        
        plt.subplot(131)
        plt.title('Episode Rewards')
        plt.plot(self.rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(132)
        sf_losses = [x[0] for x in self.losses]
        policy_losses = [x[1] for x in self.losses]
        
        plt.title('Training Losses')
        plt.plot(sf_losses, label='SF Loss')
        plt.plot(policy_losses, label='Policy Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()