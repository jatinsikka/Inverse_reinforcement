import numpy as np
from models.feature_extractor import FeatureExtractor

def load_expert_data(config, expert_buffer):
    """Load expert demonstrations from .npz file into expert buffer"""
    try:
        data = np.load(f"{config.data_dir}/{config.expert_demos_file}", allow_pickle=True)
        feature_extractor = FeatureExtractor(config.feature_dim)
        total_transitions = 0
        
        # Process each episode
        for episode_key in data.files:
            episode_data = data[episode_key]
            if isinstance(episode_data, np.ndarray) and episode_data.dtype == np.dtype('O'):
                states, actions = episode_data.item()  # Unpack the tuple stored in the array
                
                # Debug info
                print(f"\nProcessing {episode_key}")
                print(f"States shape: {states.shape if hasattr(states, 'shape') else 'no shape'}")
                print(f"First state type: {type(states[0])}")
                print(f"First state content: {states[0]}")
                
                # Process each state-action pair in the episode
                for state, action in zip(states, actions):
                    features = feature_extractor.extract(state)
                    expert_buffer.add_transition(features, action)
                    total_transitions += 1
                
                expert_buffer.end_trajectory()
        
        print(f"Loaded {total_transitions} expert transitions from {len(data.files)} episodes")
        
    except Exception as e:
        print("Error loading expert data. Keys in the .npz file:", data.files if 'data' in locals() else "No data loaded")
        raise Exception(f"Failed to load expert data: {e}")

def save_expert_data(states, actions, config):
    """Save expert demonstrations to .npz file"""
    np.savez(
        f"{config.data_dir}/{config.expert_demos_file}",
        states=states,
        actions=actions
    )

