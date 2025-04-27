import os
import numpy as np
import gymnasium as gym
import pygame
import highway_env
from gymnasium.utils.play import play
import time

# Define key mappings
key_action_map = {
    pygame.K_s: 0,  # LANE_LEFT
    pygame.K_d: 1,  # IDLE
    pygame.K_f: 2,  # LANE_RIGHT
    pygame.K_e: 3,  # FASTER
    pygame.K_q: 4,  # SLOWER
}

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def collect_human_demos(num_episodes=10):
    """Collect human demonstrations"""
    max_steps = 500  # Same as duration in config
    
    env_config = {
        'observation': {
            'type': 'Kinematics',
            'vehicles_count': 5,
            'features': ['presence', 'x', 'y', 'vx', 'vy'],
            'normalize': True
        },
        'action': {'type': 'DiscreteMetaAction'},
        'lanes_count': 3,
        'vehicles_count': 10,
        'duration': max_steps,
    }
    
    env = gym.make('highway-v0', render_mode="human", config=env_config)
    demos = []
    
    for episode in range(num_episodes):
        print(f"\nStarting Episode {episode + 1}/{num_episodes}")
        obs, _ = env.reset()
        states = []
        actions = []
        step = 0
        
        while step < max_steps:
            env.render()
            
            # Process input
            action = 1  # Default to IDLE
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:  # Early episode termination
                        step = max_steps  # Force episode end
                        break
                    if event.key in key_action_map:
                        action = key_action_map[event.key]
            
            # Execute action and record
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Flatten the observation if it's multi-dimensional
            if isinstance(obs, np.ndarray):
                obs_flat = obs.reshape(-1)  # Flatten the array
            elif isinstance(obs, dict):
                obs_flat = np.concatenate([v.flatten() for v in obs.values()])
            else:
                obs_flat = np.array(obs)
            
            states.append(obs_flat)
            actions.append(action)
            
            obs = next_obs
            step += 1
            
            if terminated or truncated:
                break
        
        # Convert lists to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        demos.append((states, actions))
        
        print(f"Episode {episode + 1} completed with {len(states)} steps")
        print(f"States shape: {states.shape}, Actions shape: {actions.shape}")
    
    return demos

def save_demos(demos, filepath):
    """Save demonstrations to file"""
    # Create a dictionary to store the demonstrations
    save_dict = {}
    
    for i, (states, actions) in enumerate(demos):
        # Store states and actions separately
        save_dict[f'states_{i}'] = states
        save_dict[f'actions_{i}'] = actions
    
    # Save to file
    np.savez(filepath, **save_dict)

def main():
    print("Starting human demonstration collection...")
    print("You will control the car to create expert demonstrations.")
    print("\nControls:")
    print("w - Lane Left")
    print("x - Lane Right")
    print("a - Accelerate")
    print("d - Decelerate")
    print("s - Do nothing")
    
    # Collect demonstrations
    demos = collect_human_demos(num_episodes=10)
    
    # Save demonstrations with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f'human_expert_demos_{timestamp}.npz'
    filepath = os.path.join(DATA_DIR, filename)
    
    # Save the demonstrations
    save_demos(demos, filepath)
    
    print("\nDemonstration collection complete!")
    print(f"Saved {len(demos)} demonstrations to {filepath}")

if __name__ == "__main__":
    main()


