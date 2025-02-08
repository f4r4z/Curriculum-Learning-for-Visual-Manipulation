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
import typing

from libero.libero import get_libero_path
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, SAC

from src.envs_gymapi import LowDimensionalObsGymEnv, AgentViewGymEnv, AgentViewGymGoalEnv, LowDimensionalObsGymGoalEnv

import src.patch

@dataclass
class Args:
    # User specific arguments
    seed: int = None
    """random seed for reproducibility"""
    video_path: str = "videos/output.mp4"
    """file path of the video output file"""
    load_path: str = "logs"
    """file path of the model file"""
    num_episodes: int = 10
    """number of episodes to generate evaluation"""

    # Environment specific arguments
    custom_bddl_path: str = None
    """if passed in, the custom path will be used for bddl file as opposed to libero default files"""
    bddl_file_name: str = "libero_90/KITCHEN_SCENE6_close_the_microwave.bddl"
    """file name of the BDDL file"""
    shaping_reward: bool = True
    """if toggled, shaping reward will be off for all goal states"""
    sparse_reward: float = 10.0
    """total sparse reward for success"""
    reward_geoms: str = None
    """if geoms are passed, those specific geoms will be rewarded, for single object predicates only [format example: ketchup_1_g1,ketchup_1_g2]"""

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

    if args.custom_bddl_path is not None:
        task_name = os.path.basename(args.custom_bddl_path)
        env_args = {
            "bddl_file_name": args.custom_bddl_path,
            "camera_heights": 128,
            "camera_widths": 128,
        }
    else:
        bddl_file_base = get_libero_path("bddl_files")
        task_name = args.bddl_file_name
        env_args = {
            "bddl_file_name": os.path.join(bddl_file_base, task_name),
            "camera_heights": 128,
            "camera_widths": 128,
        }

    # set up reward geoms
    reward_geoms = args.reward_geoms.split(",") if args.reward_geoms is not None else None

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
            envs = vec_env_class(
                [lambda: Monitor(LowDimensionalObsGymGoalEnv(**env_args)) for _ in range(args.num_envs)]
            )
        else:
            envs = vec_env_class(
                [lambda: Monitor(LowDimensionalObsGymEnv(args.shaping_reward, args.sparse_reward, reward_geoms, **env_args)) for _ in range(args.num_envs)]
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
    for i in range(250*args.num_episodes):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = envs.step(action)
        images.append(info[0]["agentview_image"])

        if dones[0]:
            count = 0
            success += 1 if info[0]["is_success"] else 0
            total_episodes += 1
            print(total_episodes)
            envs.reset()

        if total_episodes == args.num_episodes:
            break
    
        count += 1
        
    obs_to_video(images, f"{args.video_path}")
    print("# of tasks successful", success, "out of", total_episodes)
    envs.close()