import os
# add parent path to sys
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import PPO
from libero.libero.envs import SubprocVectorEnv, DummyVectorEnv, OffScreenRenderEnv
from libero.libero import get_libero_path

from src.envs import LowDimensionalObsEnv, GymVecEnvs

from IPython.display import display, HTML
from PIL import Image
import imageio
from dataclasses import dataclass

import tyro

if __name__ == "__main__":
    bddl_file_base = get_libero_path("bddl_files")
    task_name = "libero_90/KITCHEN_SCENE6_close_the_microwave.bddl"
    env_args = {
        "bddl_file_name": os.path.join(bddl_file_base, task_name),
        "camera_heights": 128,
        "camera_widths": 128,
    }

    print("setting up environment")
    envs = DummyVectorEnv(
        [lambda: OffScreenRenderEnv(**env_args) for _ in range(2)]
    )

    envs.seed(0)
    envs.reset()

    for i in range(10):
        obs, rewards, dones, info = envs.step([[0.] * 7, [0.] * 7])