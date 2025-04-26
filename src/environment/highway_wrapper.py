import gymnasium as gym
import highway_env
import numpy as np

class HighwayEnvWrapper:
    def __init__(self, config):
        self.config = config
        self.env = self._configure_env()
        
    def _configure_env(self):
        env_config = {
            'observation': self.config.observation,
            'action': self.config.action,
            'lanes_count': self.config.lanes_count,
            'initial_lane_id': 1,
            'vehicles_density': 1,
            'duration': self.config.duration,
            'simulation_frequency': 15,
            'policy_frequency': 5,
            'render_mode': 'rgb_array',
            'screen_width': 600,
            'screen_height': 150,
            'centering_position': [0.3, 0.5],
            'scaling': 5.5,
            'show_trajectories': True
        }
        
        env = gym.make('highway-v0', render_mode='rgb_array', config=env_config)
        env.reset()
        return env
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()