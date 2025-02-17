# add parent path to sys
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
import tyro
import torch
import numpy as np

from libero.libero import get_libero_path
from stable_baselines3 import PPO, SAC

import src.patch
from src.utils import setup_envs, obs_to_video
from src.args import AlgArgs, EnvArgs

@dataclass
class Args(EnvArgs, AlgArgs):
    """Note: many of the args in AlgArgs aren't actually used in this script"""

    # User specific arguments
    seed: int = None
    """random seed for reproducibility"""
    video_path: str = "video/output.mp4"
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


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.num_envs = 1

    if args.custom_bddl_path is not None:
        task_name = os.path.basename(args.custom_bddl_path)
        bddl_file = args.custom_bddl_path
    else:
        task_name = args.bddl_file_name
        bddl_file = os.path.join(get_libero_path("bddl_files"), task_name)

    print("Setting up environment")
    envs = setup_envs(bddl_file, args)

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
    last_joint_positions = []
    for i in range(args.steps_per_episode*args.num_episodes):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = envs.step(action)
        images.append(info[0]["agentview_image"])
        
        if dones[0]:
            count = 0
            success += 1 if info[0]["is_success"] else 0
            total_episodes += 1
            print(total_episodes)
            last_joint_positions.append(last_pos)
            envs.reset()
        
        last_pos = envs.envs[0].env.env.robots[0]._joint_positions

        if total_episodes == args.num_episodes:
            break
    
        count += 1
    
    obs_to_video(images, f"{args.video_path}")
    print("# of tasks successful", success, "out of", total_episodes)
    print("average of final robot joints", sum(last_joint_positions)/len(last_joint_positions))
    envs.close()