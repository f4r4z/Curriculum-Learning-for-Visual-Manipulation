import numpy as np
import gymnasium as gym
from torch import nn
import torch as th

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        if isinstance(observation_space, gym.spaces.Dict):
            n_input_channels = observation_space.get('observation').shape[0]
        else:
            n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            if isinstance(observation_space, gym.spaces.Dict):
                n_flatten = self.cnn(
                    th.as_tensor(observation_space.get("observation").sample()[None]).float()
                ).shape[1]
            else:
                n_flatten = self.cnn(
                    th.as_tensor(observation_space.sample()[None]).float()
                ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations["observation"]))
    

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class CustomCNN2(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(layer_init(nn.Linear(n_flatten, features_dim)), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
    
class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (Dict)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer of image.
    :param goal_dim: (int) Number of goal features extracted.
        This corresponds to the number of unit for the last layer of goal.
    """
    def __init__(self, observation_space, features_dim: int = 256, goal_dim: int = 32):
        super().__init__(observation_space, features_dim + goal_dim)
        extractors = {}

        for key, subspace in observation_space.items():
            if key == "observation":
                n_input_channels = subspace.shape[0]
                extractors[key] = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
            elif key == "desired_goal":
                # extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], goal_dim), nn.ReLU()) -5
                extractors[key] = nn.Linear(subspace.shape[0], goal_dim) # -2, 3

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = extractors["observation"](
                th.as_tensor(observation_space.get("observation").sample()[None]).float()
            ).shape[1]

        # 1- Neither go through linear layer
        # self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU()) # 2- CNN goes through linear layer
        self.linear = nn.Sequential(nn.Linear(n_flatten + goal_dim, features_dim + goal_dim), nn.ReLU()) # 3- Both desired goal and CNN go through linear layer
        
        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == "observation":
                # encoded_tensor_list.append(self.linear(extractor(observations[key]))) # 2-
                encoded_tensor_list.append(extractor(observations[key])) # 3-
            elif key == "desired_goal":
                encoded_tensor_list.append(extractor(observations[key]))
        # return th.cat(encoded_tensor_list, dim=1) # 2-
        return self.linear(th.cat(encoded_tensor_list, dim=1)) # 3-