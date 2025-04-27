# Inverse Reinforcement Learning for Autonomous Vehicles
# A comprehensive implementation using Highway-Env

# Find out the reward funcition based on the extracted date and the training and map it (Plot it)
# IRL
# Be clear with the concepts and how they are different IRL/imitation learning
# You can also just compare coding results 
# look into the extreacted feature code and see the dimension of the system i/O

# **** Present 40 - 45 min, lecture based 
# 1. Basic Concepts and notation (Fundamental conccepts should be completely elaborated) 
# 2. Applications 
# 3. What project I coded


# How does other people handle multi car input 
# How does spam classification work? Do they have some method handling different input systems (dimentions)? 


import numpy as np
import torch
from config.config import SFMConfig, EnvironmentConfig
from environment.highway_wrapper import HighwayEnvWrapper
from models.sfm import SFM
from utils.visualization import Visualizer
from utils.expert_data import load_expert_data
from utils.buffer import ExpertBuffer
import os

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

def train(env, agent, config, visualizer):
    total_steps = 0
    device = agent.device
    
    for episode in range(config.max_episodes):
        state, _ = env.reset()
        # Pre-process state and move to GPU at the start
        state = torch.FloatTensor(state).to(device)
        episode_reward = 0
        progress = (episode + 1) / config.max_episodes * 100
        
        for step in range(config.max_steps):
            # Get action from policy
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)
            
            # Store transition in replay buffer (keep as CPU tensors)
            agent.replay_buffer.push(
                state.cpu().numpy(), 
                action, 
                next_state.cpu().numpy(), 
                reward, 
                done
            )
            
            if len(agent.replay_buffer) > config.batch_size:
                if total_steps % config.update_frequency == 0:
                    # Batch updates are handled in update_networks
                    sf_loss, policy_loss = agent.update_networks(config.batch_size)
                    visualizer.update(episode_reward, sf_loss, policy_loss)
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            if done or truncated:
                break
        
        if episode % 10 == 0:
            print(f"Training Progress: {progress:.1f}% (Episode {episode}/{config.max_episodes}), Reward: {episode_reward}")
        
        if episode % 100 == 0:
            agent.save(f"{config.model_dir}/sfm_checkpoint_{episode}.pt")
            visualizer.save_plots(f"{config.model_dir}/training_plots_{episode}.png")
    
    print("Training Complete! (100%)")
    visualizer.plot()

def main():
    # Initialize configurations
    sfm_config = SFMConfig()
    
    # Create environment
    env = HighwayEnvWrapper(EnvironmentConfig())
    
    # Get state and action dimensions
    # The state dimension is 25 (5 features * 5 vehicles)
    state_dim = 25  # Flattened observation space
    action_dim = env.env.action_space.n
    
    # Initialize agent
    agent = SFM(state_dim, action_dim, sfm_config)
    
    # Initialize expert buffer and load demonstrations
    expert_buffer = ExpertBuffer(sfm_config.buffer_size)
    load_expert_data(sfm_config, expert_buffer)
    agent.set_expert_buffer(expert_buffer)
    
    # Initialize visualizer
    visualizer = Visualizer()
    
    # Train agent
    train(env, agent, sfm_config, visualizer)
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()





