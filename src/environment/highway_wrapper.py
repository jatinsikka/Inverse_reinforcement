import gymnasium as gym
import highway_env
import numpy as np

class HighwayEnvWrapper:
    def __init__(self, config=None):
        self.env = gym.make('highway-v0', render_mode="human")
        # Access the underlying environment using .env attribute
        config_dict = {
            'observation': {
                'type': 'Kinematics',
                'vehicles_count': config.vehicles_count if config else 5,
                'features': ['presence', 'x', 'y', 'vx', 'vy'],
                'normalize': True
            },
            'lanes_count': config.lanes_count if config else 3,
            'vehicles_count': config.vehicles_count if config else 5,
            'duration': config.duration if config else 40,
            'action': config.action if config else {'type': 'DiscreteMetaAction'}
        }
        
        self.env.unwrapped.configure(config_dict)
        
    def reset(self):
        obs, info = self.env.reset()
        return self._preprocess_observation(obs), info
    
    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        return self._preprocess_observation(next_obs), reward, terminated, truncated, info
    
    def _preprocess_observation(self, obs):
        """Preprocess the observation to flatten it into the correct shape"""
        if isinstance(obs, np.ndarray):
            return obs.reshape(-1)  # Flatten to 1D array
        return obs
    
    def close(self):
        self.env.close()


