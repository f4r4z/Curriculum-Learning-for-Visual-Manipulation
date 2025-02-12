# add parent path to sys so we can reference src
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
from dataclasses import dataclass
import tyro
import wandb
import torch
import numpy as np
from typing import List, Tuple, Optional

from libero.libero import get_libero_path
from stable_baselines3.common.callbacks import CheckpointCallback
from src.callbacks import VideoWriter

from utils import WandbArgs, AlgArgs, EnvArgs, setup_envs, setup_model

@dataclass
class Args(WandbArgs, AlgArgs, EnvArgs):
    # User specific arguments
    seed: Optional[int] = None
    """random seed for reproducibility"""
    video_path: str = "videos/output.mp4"
    """file path of the video output file"""
    save_path: str = "event_logs"
    """file path of the model output file"""
    load_path: Optional[str] = None # "models/checkpoints/"
    """directory path of the models checkpoints"""
    model_path: Optional[str] = None
    """path to existing model if loading a model"""
    verbose: Optional[int] = 1
    """verbosity of outputs, with 0 being least"""

    # Environment specific arguments
    custom_bddl_path: Optional[str] = None
    """if passed in, the custom path will be used for bddl file as opposed to libero default files"""
    bddl_file_name: str = "libero_90/KITCHEN_SCENE6_close_the_microwave.bddl"
    """file name of the BDDL file"""


if __name__ == "__main__":
    args = tyro.cli(Args)


    if args.custom_bddl_path is not None:
        task_name = os.path.basename(args.custom_bddl_path)
        bddl_file = args.custom_bddl_path
    else:
        task_name = args.bddl_file_name
        bddl_file = os.path.join(get_libero_path("bddl_files"), task_name)
    

    print("Setting up environment")
    envs = setup_envs(bddl_file, args, verbose=args.verbose)

    # Seeding everything
    if args.seed is not None:
        envs.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)


    print("Creating save directory")
    run_name = os.path.join(
        task_name,
        f"{args.get_alg_str()}_seed_{args.seed}",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    save_path = os.path.join(args.save_path, run_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            dir=save_path,
            config=vars(args),
            sync_tensorboard=True,
            name=run_name
        )


    print("Setting up model")
    model = setup_model(args, envs, args.seed, save_path, args.model_path)
    log_interval = 1 if args.alg == "ppo" else 2
    # device = args.get_device()


    print("Setting up callbacks")
    callbacks = []

    # checkpoint callback
    callbacks.append(CheckpointCallback(save_freq=log_interval*1024, save_path=save_path, name_prefix="model"))

    # log videos
    callbacks.append(VideoWriter(n_steps=5000 * args.num_envs))

    if args.exploration_alg != None:
        callbacks.append(args.get_exploration_callback(envs))
    

    print("Start training")
    model.learn(total_timesteps=args.total_timesteps, log_interval=log_interval, callback=callbacks, progress_bar=False)
    model.save(save_path)

    del model