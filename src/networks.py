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
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
            elif key == "desired_goal":
                # extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], goal_dim), nn.ReLU()) # -5
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
    

class CustomCombinedExtractor2(BaseFeaturesExtractor):
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
                self.conv = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
            elif key == "desired_goal":
                n_input_goals = subspace.shape[0]

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.conv(
                th.as_tensor(observation_space.get("observation").sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten + n_input_goals, features_dim + goal_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations["observation"]  = self.conv(observations["observation"])
        return self.linear(th.cat([observations["observation"], observations["desired_goal"]], dim=1))
    

class CustomCombinedPatchExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (Dict)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer of image.
    :param goal_dim: (int) Number of goal features extracted.
        This corresponds to the number of unit for the last layer of goal.
    """
    def __init__(self, observation_space, patch_size=[16,16], embed_size=64, no_patch_embed_bias=False, features_dim: int = 256, goal_dim: int = 32):
        super().__init__(observation_space, features_dim + goal_dim)
        extractors = {}
        C, H, W = observation_space["observation"].shape[0], observation_space["observation"].shape[1], observation_space["observation"].shape[2]
        num_patches = (H // patch_size[0] // 2) * (W // patch_size[1] // 2)
        self.img_size = (H, W)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.h, self.w = H // patch_size[0] // 2, W // patch_size[1] // 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                C, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            ),
            nn.BatchNorm2d(
                64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(
            64,
            embed_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )
        self.bn = nn.BatchNorm2d(embed_size)
        self.flatten = nn.Flatten()
        
        self.desired_layer = nn.Linear(observation_space["desired_goal"].shape[0], goal_dim)

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.flatten(self.bn(self.proj(self.conv((
                th.as_tensor(observation_space.get("observation").sample()[None]).float()
            ))))).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten + goal_dim, features_dim + goal_dim), nn.ReLU())


    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations["observation"] = self.conv(observations["observation"])
        observations["observation"] = self.proj(observations["observation"])
        observations["observation"] = self.bn(observations["observation"])
        observations["observation"] = self.flatten(observations["observation"])

        observations["desired_goal"] = self.desired_layer(observations["desired_goal"])

        return self.linear(th.cat([observations["observation"], observations["desired_goal"]], dim=1))