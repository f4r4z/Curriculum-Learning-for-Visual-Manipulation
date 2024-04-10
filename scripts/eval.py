# add parent path to sys
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
import tyro
import imageio
from IPython.display import HTML
import torch
import numpy as np

from libero.libero import get_libero_path
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, SAC

from src.envs_gymapi import LowDimensionalObsGymEnv, AgentViewGymEnv, AgentViewGymGoalEnv

@dataclass
class Args:
    # User specific arguments
    seed: int = None
    """random seed for reproducibility"""
    video_path: str = "videos/output.mp4"
    """file path of the video output file"""
    load_path: str = "logs"
    """file path of the model file"""

    # Environment specific arguments
    bddl_file_name: str = "libero_90/KITCHEN_SCENE6_close_the_microwave.bddl"
    """file name of the BDDL file"""

    # Algorithm specific arguments
    alg: str = "ppo"
    """algorithm to use for training"""
    her: bool = False
    """if toggled, SAC will use HER otherwise it would not"""
    num_envs: int = 1
    """number of LIBERO environments"""
    visual_observation: bool = False
    """if toggled, the environment will return visual observation otherwise it would not"""

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
    args.num_envs = 1

    bddl_file_base = get_libero_path("bddl_files")
    task_name = "libero_90/KITCHEN_SCENE6_close_the_microwave.bddl"
    env_args = {
        "bddl_file_name": os.path.join(bddl_file_base, task_name),
        "camera_heights": 128,
        "camera_widths": 128,
    }

    print("Setting up environment")
    vec_env_class = DummyVecEnv
    if args.visual_observation:
        if args.her:
            envs = vec_env_class(
                [lambda: Monitor(AgentViewGymGoalEnv(**env_args)) for _ in range(args.num_envs)]
            )
        else:
            envs = vec_env_class(
                [lambda: Monitor(AgentViewGymEnv(**env_args)) for _ in range(args.num_envs)]
            )
    else:
        if args.her:
            raise ValueError("HER is only supported for visual observation")
        else:
            envs = vec_env_class(
                [lambda: Monitor(LowDimensionalObsGymEnv(**env_args)) for _ in range(args.num_envs)]
            )

    # Seeding everything
    if args.seed is not None:
        envs.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.alg == "ppo":
        algorithm = PPO
    elif args.alg == "sac":
        algorithm = SAC

    # start evaluation
    print("loading model")
    model = algorithm.load(f"{args.load_path}", env=envs if args.her else None)

    obs = envs.reset()

    images = []

    print("generating video")
    count = 0
    success = 0
    total_episodes = 0
    for i in range(250*10):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = envs.step(action)
        images.append(info[0]["agentview_image"])

        if dones[0] or count >= 250:
            count = 0
            success += dones[0]
            total_episodes += 1
            print(total_episodes)
            envs.reset()

        if total_episodes == 10:
            break
    
        count += 1
        
    obs_to_video(images, f"{args.video_path}")
    print("# of tasks successful", success)
    envs.close()