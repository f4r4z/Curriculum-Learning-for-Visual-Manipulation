import os
# add parent path to sys
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from libero.libero.envs import SubprocVectorEnv, OffScreenRenderEnv, DummyVectorEnv
from libero.libero import get_libero_path

from src.envs import LowDimensionalObsEnv, AgentViewEnv, AgentViewGoalEnv, GymVecEnvs
from src.callbacks import TensorboardCallback

import gymnasium as gym

from IPython.display import display, HTML
from PIL import Image
import imageio
from dataclasses import dataclass

import torch as th
import torch.nn as nn

import numpy as np
import tyro


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
        print("input channels: ", n_input_channels)
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
        return self.linear(self.cnn(observations))
    
@dataclass
class Args:

    video_path: str = "videos/output.mp4"
    """file path of the video output file"""
    model_path: str = "models/close_the_microwave"
    """file path of the model output file"""
    checkpoints_path: str = "models/checkpoints/"
    """directory path of the models checkpoints"""
    # Algorithm specific arguments
    train: bool = False
    """if toggled, model will train otherwise it would not"""
    eval: bool = True
    """if toggled, model will load and evaluate a model otherwise it would not"""
    total_timesteps: int = 250000
    """total timesteps of the experiments"""
    learning_rate: float = 0.0003
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """number of LIBERO environments"""
    save_freq: int = 10000
    "save frequency of model checkpoint during training"

def obs_to_video(images, filename):
    """
    converts a list of images to video and writes the file
    """
    video_writer = imageio.get_writer(filename, fps=60)
    for image in images:
        video_writer.append_data(image[::-1])
    video_writer.close()
    HTML("""
        <video width="640" height="480" controls>
            <source src="output.mp4" type="video/mp4">
        </video>
        <script>
            var video = document.getElementsByTagName('video')[0];
            video.playbackRate = 2.0; // Increase the playback speed to 2x
            </script>    
    """)

if __name__ == "__main__":
    args = tyro.cli(Args)
    bddl_file_base = get_libero_path("bddl_files")
    task_name = "libero_90/KITCHEN_SCENE6_close_the_microwave.bddl"
    env_args = {
        "bddl_file_name": os.path.join(bddl_file_base, task_name),
        "camera_heights": 128,
        "camera_widths": 128,
    }

    print("setting up environment")

    '''
    envs = SubprocVectorEnv(
        [lambda: LowDimensionalObsEnv(**env_args) for _ in range(args.num_envs)]
    )
    '''
    
    envs = DummyVectorEnv(
        [lambda: AgentViewGoalEnv(**env_args) for _ in range(args.num_envs)]
    )
    envs = GymVecEnvs(envs)
    # import ipdb; ipdb.set_trace()

    # Create the agent with two layer of 128 units
    if args.train:
        print("training with learning rate bruh: ", args.learning_rate)
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )
        checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=args.checkpoints_path, name_prefix="pulisic_her_sac_agentview_model")
        tensorboard_callback = TensorboardCallback()
        model = SAC("MultiInputPolicy", envs, verbose=1, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future',), policy_kwargs=policy_kwargs, learning_rate=args.learning_rate, tensorboard_log="./logs")
        model.learn(total_timesteps=args.total_timesteps, log_interval=1, progress_bar=True, callback=[checkpoint_callback, tensorboard_callback])
        model.save(f"{args.model_path}")

        del model

    if args.eval:
        print("loading model")
        model = SAC.load(f"{args.model_path}")

        obs = envs.reset()

        # second environment for visualization
        off_env = OffScreenRenderEnv(**env_args)
        off_env.reset()
        images = []

        for i in range(500):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = envs.step(action)

            # for visualization
            off_obs, _, _, _, = off_env.step(action[0])
            images.append(off_obs["agentview_image"])
            


        obs_to_video(images, f"{args.video_path}")
        off_env.close()
        envs.close()