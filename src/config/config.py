from dataclasses import dataclass

@dataclass
class SFMConfig:
    # Environment
    env_name: str = 'highway-v0'
    max_episodes: int = 1000
    max_steps: int = 500
    
    # SFM Parameters
    feature_dim: int = 5
    hidden_dim: int = 512  # Increased from 256
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 0.001
    
    # Training
    batch_size: int = 256  # Increased from 64
    buffer_size: int = 100000
    update_frequency: int = 50  # Decreased from 100 for more frequent updates
    
    # Paths
    model_dir: str = 'models'
    data_dir: str = 'data'
    expert_demos_file: str = 'human_expert_demos_20250426_191310.npz'

@dataclass
class EnvironmentConfig:
    observation: dict = None
    action: dict = None
    lanes_count: int = 3
    vehicles_count: int = 5
    duration: int = 40
    
    def __post_init__(self):
        self.observation = {
            'type': 'Kinematics',
            'vehicles_count': self.vehicles_count,
            'features': ['presence', 'x', 'y', 'vx', 'vy'],
            'normalize': True
        }
        self.action = {
            'type': 'DiscreteMetaAction'
        }

