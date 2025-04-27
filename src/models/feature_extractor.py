import numpy as np
import torch

class FeatureExtractor:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim

    def extract(self, state):
        """
        Extract features from state observation
        state: numpy array or dictionary containing observation from highway-env
        returns: numpy array of features
        """
        try:
            # If state is already a numpy array with correct dimension
            if isinstance(state, np.ndarray):
                if state.shape[0] == self.feature_dim:
                    return state
                else:
                    # Pad or truncate to match feature_dim
                    features = np.zeros(self.feature_dim)
                    features[:min(self.feature_dim, state.shape[0])] = state[:min(self.feature_dim, state.shape[0])]
                    return features
            
            # If state is a dictionary (from highway-env)
            elif isinstance(state, dict):
                # Extract relevant features from the dictionary
                # Modify this based on your environment's state structure
                return np.zeros(self.feature_dim)  # Placeholder
            
            # If state is a string
            elif isinstance(state, str):
                try:
                    # Try to convert string to numpy array
                    values = np.fromstring(state, sep=' ')
                    features = np.zeros(self.feature_dim)
                    features[:min(self.feature_dim, values.shape[0])] = values[:min(self.feature_dim, values.shape[0])]
                    return features
                except:
                    print(f"Warning: Could not parse state string: {state}")
                    return np.zeros(self.feature_dim)
            
            else:
                print(f"Warning: Unknown state type: {type(state)}")
                return np.zeros(self.feature_dim)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(self.feature_dim)



