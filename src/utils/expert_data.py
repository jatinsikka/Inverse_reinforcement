import numpy as np
import os
from models.feature_extractor import FeatureExtractor

def load_expert_data(config, expert_buffer):
    """Load expert demonstrations from file"""
    filepath = os.path.join(config.data_dir, config.expert_demos_file)
    total_transitions = 0
    
    try:
        data = np.load(filepath, allow_pickle=True)
        num_episodes = len(data.files) // 2  # Since we store states and actions separately
        
        for i in range(num_episodes):
            states = data[f'states_{i}']
            actions = data[f'actions_{i}']
            
            print(f"\nProcessing episode {i}")
            print(f"States shape: {states.shape}")
            print(f"Actions shape: {actions.shape}")
            
            # Process each state-action pair in the episode
            for state, action in zip(states, actions):
                expert_buffer.add_transition(state, action)
                total_transitions += 1
            
            expert_buffer.end_trajectory()
        
        print(f"Loaded {total_transitions} expert transitions from {num_episodes} episodes")
        
        if total_transitions == 0:
            raise Exception("No valid transitions were loaded")
        
    except Exception as e:
        print("Error loading expert data:", str(e))
        raise

def save_expert_data(states, actions, config):
    """Save expert demonstrations to .npz file"""
    np.savez(
        f"{config.data_dir}/{config.expert_demos_file}",
        states=states,
        actions=actions
    )





