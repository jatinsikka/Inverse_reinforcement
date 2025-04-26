import os
import numpy as np
import gymnasium as gym
import pygame
import highway_env
from gymnasium.utils.play import play
import time

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def collect_human_demos(num_episodes=10):
    """Collect human demonstrations"""
    max_steps = 500  # Same as duration in config
    
    env_config = {
        'observation': {
            'type': 'Kinematics',
            'vehicles_count': 5,
            'features': ['presence', 'x', 'y', 'vx', 'vy', 'heading'],
        },
        'action': {'type': 'DiscreteMetaAction'},
        'lanes_count': 3,
        'vehicles_count': 10,
        'duration': max_steps,  # [s]
        'simulation_frequency': 15,
        'policy_frequency': 5,
        'screen_width': 1000,
        'screen_height': 200,
        'centering_position': [0.3, 0.5],
        'scaling': 5.5,
        'show_trajectories': True,
    }
    
    # Create environment with human rendering
    env = gym.make('highway-v0', render_mode="human", config=env_config)
    demos = []
    
    print(f"\nStarting collection of {num_episodes} episodes")
    print("\nControls:")
    print("w - Lane Left")
    print("x - Lane Right")
    print("a - Accelerate")
    print("d - Decelerate")
    print("s - Do nothing")
    print("q - End current episode early")
    print(f"\nEach episode will automatically end after {max_steps} steps")
    
    # Define key mappings
    key_action_map = {
        pygame.K_w: 0,  # LANE_LEFT
        pygame.K_s: 1,  # IDLE
        pygame.K_x: 2,  # LANE_RIGHT
        pygame.K_a: 3,  # FASTER
        pygame.K_d: 4,  # SLOWER
    }
    
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
            states.append(obs)
            actions.append(action)
            
            obs = next_obs
            step += 1
            
            if terminated or truncated:
                break
        
        if len(states) > 0:
            demos.append((np.array(states), np.array(actions)))
            print(f"Episode {episode + 1} completed with {len(states)} steps")
        else:
            print(f"Episode {episode + 1} skipped (no data collected)")
    
    env.close()
    print(f"\nCollection complete! Collected {len(demos)} episodes.")
    return demos

def save_demos(demos, filename):
    """Save demonstrations to file"""
    # Save each episode separately
    save_dict = {}
    for i, (states, actions) in enumerate(demos):
        # Create a tuple of states and actions without trying to combine them
        save_dict[f'episode_{i}'] = {
            'states': states,  # Shape: (timesteps, vehicles, features)
            'actions': actions  # Shape: (timesteps,)
        }
    
    np.savez(filename, **save_dict)
    total_transitions = sum(len(states) for states, _ in demos)
    print(f"Saved {total_transitions} transitions from {len(demos)} episodes to {filename}")

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



