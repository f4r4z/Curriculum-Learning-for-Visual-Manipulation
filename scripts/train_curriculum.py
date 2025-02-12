# add parent path to sys
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
START_TIME = datetime.datetime.now()
START_TIME_STR = START_TIME.strftime("%Y%m%d-%H%M%S")

from dataclasses import dataclass
import tyro
import wandb
import torch
import numpy as np
from typing import List, Tuple, Optional

from stable_baselines3.common.callbacks import CheckpointCallback

from src.callbacks import VideoWriter, StopTrainingOnSuccessRateThreshold

from utils import WandbArgs, AlgArgs, EnvArgs, setup_envs, setup_model

import inspect

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
    curriculum_file: str
    """The path to a python file containing functions that generate BDDL files"""

    n_eval_episodes: int = 10
    """number of episodes to run for evaluation and success rate checking. Set to 0 to not eval and always train to total_timesteps"""
    success_rate_threshold: float = 0.7
    """success rate to reach before moving on to the next subtask of the curriculum"""


def load_bddls(curriculum_file: str):
    assert curriculum_file is not None
    assert os.path.exists(curriculum_file)
    with open(curriculum_file, 'r') as f:
        curriculum_file_str = f.read()
    
    namespace = {}
    exec(curriculum_file_str, namespace)

    bddls: List[Tuple[str, str]] = []
    for k, func in namespace.items():
        if not inspect.isfunction(func) or inspect.isbuiltin(func): # ignore non-functions and builtins
            print(f"skipping {k} because it is builtin or is not a function")
            continue
        if k.startswith("__") and k.endswith("__"): # ignore functions named __name__
            print(f"skipping {k} because it is in the format __name__")
            continue
        if len(inspect.signature(func).parameters) == 0:
            bddl = func()
            if type(bddl) is str:
                bddls.append((func.__name__, bddl))
            elif type(bddl) is list:
                for i, s in enumerate(bddl):
                    assert type(s) is str
                    bddls.append((f"{func.__name__}_{i}", s))
    # for k, bddl in bddls:
    #     print(k)
    #     print(bddl)
    return bddls


def create_envs(bddl_str: str, args: Args, tmp_dir = "."):
    bddl_path = os.path.join(tmp_dir, f"tmp_bddl_{START_TIME_STR}.bddl")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    with open(bddl_path, 'w') as f:
        f.write(bddl_str)

    envs = setup_envs(bddl_path, args, verbose=args.verbose)
    
    os.remove(bddl_path)
    return envs


if __name__ == "__main__":
    args = tyro.cli(Args)


    print("Loading bddls")
    bddls = load_bddls(args.curriculum_file)
    assert len(bddls) > 0


    print("Creating save directory")
    task_name = os.path.splitext(os.path.basename(args.curriculum_file))[0]
    run_name = os.path.join(task_name, f"{args.get_alg_str()}_seed_{args.seed}", START_TIME_STR)
    save_path = os.path.join(args.save_path, run_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tensorboard_path = os.path.join(save_path, "tensorboard")
    models_path = os.path.join(save_path, "models")
    checkpoints_path = os.path.join(save_path, "checkpoints")
    tmp_path = os.path.join(save_path, "tmp")

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            dir=save_path,
            config=vars(args),
            sync_tensorboard=True,
            name=run_name
        )


    print("Setting up environment")
    # Create temporary env using first bddls because model requires an env to initialize
    envs = create_envs(bddls[0][1], args, tmp_dir=tmp_path)

    # Seeding everything
    if args.seed is not None:
        envs.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)


    print("Setting up model")
    model = setup_model(args, envs, args.seed, save_path, args.model_path)
    log_interval = 1 if args.alg == "ppo" else 2
    # device = args.get_device()


    print("Start training")
    for i, (subtask_name, bddl) in enumerate(bddls):
        print(f"Starting subtask {i+1}/{len(bddls)} ({subtask_name}) at step {model.num_timesteps}")


        envs = create_envs(bddl, args, tmp_dir=tmp_path)
        if args.seed is not None:
            envs.seed(args.seed)
        model.set_env(envs)


        callbacks = []

        # checkpoint callback
        callbacks.append(CheckpointCallback(save_freq=log_interval*32, save_path=checkpoints_path, name_prefix="model"))

        # log videos
        callbacks.append(VideoWriter(n_steps=5000 * args.num_envs))

        # Stop training when the model reaches the success rate threshold
        if i < len(bddls)-1: # on the last subtask, train all the way to the end
            callbacks.append(StopTrainingOnSuccessRateThreshold(threshold=args.success_rate_threshold))

        # exploration technique callbacks
        if args.exploration_alg != None:
            callbacks.append(args.get_exploration_callback(envs))
        
        
        # reset these buffers so the training stats for the previous subtask doesn't leak into this subtask
        model.ep_info_buffer = None
        model.ep_success_buffer = None

        model.learn(
            total_timesteps=args.total_timesteps,
            tb_log_name=f"{i}_{subtask_name}",
            log_interval=log_interval,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=False
        )


        if args.wandb: # save tensorboard files to wandb
            wandb.save(os.path.join(tensorboard_path, "*", "*"), base_path=tensorboard_path, policy='now')
        model.save(os.path.join(models_path, f"{i}_{subtask_name}"))
        if args.wandb: # save models to wandb
            wandb.save(os.path.join(models_path, f"{i}_{subtask_name}.zip"), base_path=save_path, policy='now')

        envs.close()

    del model