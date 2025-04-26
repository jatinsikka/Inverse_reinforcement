import numpy as np
import torch

class FeatureExtractor:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim

    def extract(self, state):
        """
        Extract features from state observation
        state: dictionary containing observation from highway-env
        returns: numpy array of features
        """
        try:
            # Initialize feature vector
            features = np.zeros(self.feature_dim)
            
            # Print state format for debugging
            print("State type:", type(state))
            print("State content:", state)
            
            # Extract ego vehicle state
            if isinstance(state, dict):
                # If state is a dictionary (newer format)
                ego_vehicle = state['ego_vehicle']
                other_vehicles = state['other_vehicles']
            elif isinstance(state, np.ndarray):
                # If state is a numpy array (older format)
                # Assuming first row is ego vehicle and rest are other vehicles
                ego_vehicle = state[0]
                other_vehicles = state[1:]
            else:
                raise ValueError(f"Unexpected state format: {type(state)}")

            # Extract ego vehicle features (position, velocity, heading)
            features[0:2] = ego_vehicle[0:2]  # position
            features[2:4] = ego_vehicle[2:4]  # velocity
            if len(ego_vehicle) > 4:
                features[4] = ego_vehicle[4]   # heading

            # Extract features from other vehicles
            if len(other_vehicles) > 0:
                # Calculate relative positions and velocities
                relative_pos = other_vehicles[:, 0:2] - ego_vehicle[0:2]
                relative_vel = other_vehicles[:, 2:4] - ego_vehicle[2:4]
                
                # Take the closest vehicle's features
                closest_idx = np.argmin(np.linalg.norm(relative_pos, axis=1))
                features[5:7] = relative_pos[closest_idx]
                features[7:9] = relative_vel[closest_idx]
                if other_vehicles.shape[1] > 4:
                    features[9] = other_vehicles[closest_idx, 4]  # heading

            return features

        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            raise e
