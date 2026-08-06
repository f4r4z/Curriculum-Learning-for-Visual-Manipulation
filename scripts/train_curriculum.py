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
import copy
import gc

from stable_baselines3.common.callbacks import CheckpointCallback

from src.callbacks import VideoWriter, StopTrainingOnSuccessRateThreshold
from src.utils import setup_envs, setup_run_at_path, setup_model, get_open_files_count
from src.args import WandbArgs, AlgArgs, EnvArgs

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

    curriculum_file: str = tyro.MISSING
    """The path to a python file containing functions that generate BDDL files"""
    ignore_tasks: str = ""
    """Comma separated list of task names(function names) to ignore when reading the curriculum file"""
    ignore_until: str = ""
    """Skip all tasks until reaching a task with this name (includes the number)"""
    success_rate_threshold: float = 0.7
    """success rate to reach before moving on to the next subtask of the curriculum"""
    final_task_timesteps: Optional[int] = None
    """The number of timesteps to run the final task. If not set, will equal total_timesteps"""


def load_bddls(curriculum_file: str, ignore_until: str = "", ignore_tasks: List[str] = []):
    assert curriculum_file is not None
    assert os.path.exists(curriculum_file)
    with open(curriculum_file, 'r') as f:
        curriculum_file_str = f.read()
    
    namespace = {}
    exec(curriculum_file_str, namespace)

    bddls: List[Tuple[str, str]] = []
    for k, func in namespace.items():
        if not inspect.isfunction(func) or inspect.isbuiltin(func): # ignore non-functions and builtins
            # print(f"skipping {k} because it is builtin or is not a function")
            continue
        if k.startswith("__") and k.endswith("__"): # ignore functions named __name__
            # print(f"skipping {k} because it is in the format __name__")
            continue
        if k in ignore_tasks: 
            print(f"skipping {k} in ignore_tasks")
            continue
        if ignore_until != "" and not ignore_until.startswith(k):
            print(f"skipping {k} until ignore_until")
            continue
        if len(inspect.signature(func).parameters) == 0:
            bddl = func()
            if type(bddl) is str:
                bddl_name = func.__name__
                if bddl_name == ignore_until:
                    ignore_until = ""
                bddls.append((bddl_name, bddl))
                print(f"Added single task '{bddl_name}'")
            elif type(bddl) is list:
                count = 0
                for i, s in enumerate(bddl):
                    assert type(s) is str
                    bddl_name = f"{func.__name__}_{i}"
                    if ignore_until != "": # ignore until this if we're skipping tasks
                        if bddl_name != ignore_until: 
                            print(f"skipping {bddl_name} until ignore_until")
                            continue
                        else:
                            ignore_until = ""
                    bddls.append((f"{func.__name__}_{i}", s))
                    count += 1
                print(f"Added task space '{func.__name__}' with {count} steps")
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
    bddls = load_bddls(args.curriculum_file, args.ignore_until, [t.strip() for t in args.ignore_tasks.split(',')])
    assert len(bddls) > 0


    print("Creating save directory")
    task_name = os.path.splitext(os.path.basename(args.curriculum_file))[0]
    run_name, save_path = setup_run_at_path(
        args.save_path,
        task_name,
        f"{args.get_alg_str()}_seed_{args.seed}",
        START_TIME_STR
    )
    tensorboard_path = os.path.join(save_path, "tensorboard")
    models_path = os.path.join(save_path, "models")
    checkpoints_path = os.path.join(save_path, "checkpoints")
    tmp_path = os.path.join(save_path, "tmp")


    print("Verifying bddls")
    verify_args = copy.copy(args)
    verify_args.num_envs = 1
    for i, (subtask_name, bddl) in enumerate(bddls):
        try:
            envs = create_envs(bddl, verify_args, tmp_dir=tmp_path)
            envs.reset()
            envs.step(np.array([envs.action_space.sample()]))
            envs.close()
        except Exception as e:
            print(f"Exception in bddl {i} '{subtask_name}'")
            raise e


    args.init_wandb_if_toggled(
        dir=save_path,
        config=vars(args),
        sync_tensorboard=True,
        name=run_name
    )


    # Create temporary env using first bddl for the model to use in initialization
    envs = create_envs(bddls[0][1], args, tmp_dir=tmp_path)


    # Seeding everything
    if args.seed is not None:
        envs.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)


    print("Setting up model")
    model = setup_model(args, envs, args.seed, save_path, tensorboard_path, args.model_path)
    log_interval = 1 if args.alg == "ppo" else 2
    # device = args.get_device()


    envs.close()
    del envs
    print("Start training")
    for i, (subtask_name, bddl) in enumerate(bddls):
        print(f"Starting subtask {i+1}/{len(bddls)} ({subtask_name}) at step {model.num_timesteps}")
        print("Open files before subtask:", get_open_files_count())
        is_final_task = i == len(bddls)-1

        envs = create_envs(bddl, args, tmp_dir=tmp_path)
        print("Open files after create_envs:", get_open_files_count())
        if args.seed is not None:
            envs.seed(args.seed)
        model.set_env(envs)


        callbacks = []

        # checkpoint callback
        callbacks.append(CheckpointCallback(save_freq=log_interval*32, save_path=checkpoints_path, name_prefix="model"))

        # log videos
        callbacks.append(VideoWriter(n_steps=5000 * args.num_envs))

        # Stop training when the model reaches the success rate threshold
        if not is_final_task: # on the last subtask, train all the way to the end
            callbacks.append(StopTrainingOnSuccessRateThreshold(
                threshold=args.success_rate_threshold, 
                min_count=log_interval*args.num_envs,
            ))

        # exploration technique callbacks
        if args.exploration_alg is not None:
            callbacks.append(args.get_exploration_callback(envs))
        
        
        # reset these buffers so the training stats for the previous subtask doesn't leak into this subtask
        model.ep_info_buffer = None
        model.ep_success_buffer = None

        total_timesteps = args.total_timesteps
        if is_final_task and args.final_task_timesteps is not None:
            total_timesteps = args.final_task_timesteps

        model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=f"{i}_{subtask_name}",
            log_interval=log_interval,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=False
        )
        print("Open files after learn:", get_open_files_count())


        if args.wandb: # save tensorboard files to wandb
            wandb.save(os.path.join(tensorboard_path, "*", "*"), base_path=tensorboard_path, policy='now')
        model.save(os.path.join(models_path, f"{i}_{subtask_name}"))
        if args.wandb: # save models to wandb
            wandb.save(os.path.join(models_path, f"{i}_{subtask_name}.zip"), base_path=save_path, policy='now')

        print("Open files before close:", get_open_files_count())
        envs.close()
        del envs
        gc.collect()
        print("Open files after close:", get_open_files_count())

    del model