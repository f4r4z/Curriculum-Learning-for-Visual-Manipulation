import os
import time
from typing import List, Optional, Tuple

import imageio
from IPython.display import HTML

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, SAC

from . import args
from .envs_gymapi import LowDimensionalObsGymEnv, LowDimensionalObsGymGoalEnv, AgentViewGymEnv, AgentViewGymGoalEnv
from .networks import CustomCNN, CustomCombinedPatchExtractor
from .her_replay_buffer_modified import HerReplayBufferModified

import subprocess
import multiprocessing
create_env_err_count = 0

def obs_to_video(images: list, filename: str):
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

def get_open_files_count():
    output = subprocess.check_output(['lsof', '-w', '-Ff', '-p', str(os.getpid())])
    num_fds = 0
    for line in output.decode().split('\n'):
        if line.startswith('f') and line[1:].isdigit():
            num_fds += 1
    return num_fds

class EnvAndAlgArgs(args.EnvArgs, args.AlgArgs):
    pass

def setup_envs(
    bddl_file: str,
    args: EnvAndAlgArgs,
    **env_args_override
) -> VecEnv:
    print("Setting up environment")

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    
    if not args.truncate:
        env_args["horizon"] = args.total_timesteps
        
    env_args.update(env_args_override)

    # vec_env_class = SubprocVecEnv if args.num_envs > 1 else DummyVecEnv
    if args.visual_observation:
        if args.her:
            envs = [lambda: Monitor(AgentViewGymGoalEnv(**env_args), info_keywords=["is_success"]) for _ in range(args.num_envs)]
        else:
            envs = [lambda: Monitor(AgentViewGymEnv(**env_args), info_keywords=["is_success"]) for _ in range(args.num_envs)]
    else:
        if args.her:
            envs = [lambda: Monitor(LowDimensionalObsGymGoalEnv(**env_args), info_keywords=["is_success"]) for _ in range(args.num_envs)]
        else:
            envs = [lambda: Monitor(LowDimensionalObsGymEnv(
                args.shaping_reward,
                args.sparse_reward,
                reward_geoms=args.reward_geoms.split(",") if args.reward_geoms is not None else None,
                dense_reward_multiplier=args.dense_reward_multiplier,
                steps_per_episode=args.steps_per_episode,
                sim_states=args.fetch_sim_states(),
                setup_demo=args.fetch_setup_demo(),
                **env_args
            ), info_keywords=["is_success"]) for _ in range(args.num_envs)]
    
    if args.num_envs > 1:
        while True:
            global create_env_err_count
            try:
                print("Open files before SubprocVecEnv:", get_open_files_count())
                env = SubprocVecEnv(envs, start_method=args.multiprocessing_start_method)
                print("Open files after SubprocVecEnv:", get_open_files_count())
                create_env_err_count = 0
                return env
            except OSError as e:
                create_env_err_count += 1
                print(f"Got error while creating envs, trying again in {create_env_err_count}s: {e}")
                print(f"processes: {len(multiprocessing.active_children())}")
                time.sleep(create_env_err_count)
                
    else:
        print("Open files before DummyVecEnv:", get_open_files_count())
        env = DummyVecEnv(envs)
        print("Open files after DummyVecEnv:", get_open_files_count())
        return env


def setup_run_at_path(base_path: str, *paths: str):
    run_name = os.path.join(*paths)
    save_path = os.path.join(base_path, run_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return run_name, save_path


def setup_model(
    args: args.AlgArgs, 
    env: VecEnv,
    seed, 
    save_path: str, 
    tensorboard_path: Optional[str] = None,
    load_path: Optional[str] = None
):
    if tensorboard_path == None:
        tensorboard_path = save_path

    if args.visual_observation:
        policy_kwargs = dict(
            features_extractor_class=CustomCombinedPatchExtractor if args.her else CustomCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )
        policy_class = "MultiInputPolicy" if args.her else "CnnPolicy"
    else:
        policy_kwargs = dict(net_arch=[128, 128])
        policy_class = "MultiInputPolicy" if args.her else "MlpPolicy"
        
    algorithm = None
    if args.alg == "ppo":
        algorithm = PPO
        model = PPO(
            policy_class,
            env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            tensorboard_log=tensorboard_path,
            n_steps=args.n_steps,
            ent_coef=args.ent_coef,
            # clip_range=args.clip_range,
            seed=seed
        )
    elif args.alg == "sac":
        algorithm = SAC
        if args.her:
            model = SAC(
                policy_class,
                env,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tensorboard_path,
                seed=seed,
                learning_rate=args.learning_rate,
                learning_starts=1000*env.num_envs,
                batch_size=256,
                train_freq=(1, "step"),
                gradient_steps=-1,
                replay_buffer_class=HerReplayBufferModified,
                replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future',)
            )
        else:
            model = SAC(
                policy_class,
                env,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tensorboard_path,
                seed=seed,
                learning_rate=args.learning_rate,
                learning_starts=1000*env.num_envs,
                batch_size=256,
                train_freq=(1, "step"),
                gradient_steps=-1,
            )
    else:
        raise ValueError(f"Algorithm {args.alg} is not in supported list [ppo, sac]")

    if load_path != None:
        print("loading model from ", load_path)
        model = algorithm.load(
            f"{args.model_path}",
            env=env,
            policy=policy_class,
            verbose=1,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            tensorboard_log=tensorboard_path,
            n_steps=args.n_steps,
            ent_coef=args.ent_coef,
            # clip_range = args.clip_range,
            seed=seed,
        )
        # model = algorithm.load(f"{load_path}", env=env)
        # model.learning_rate = args.learning_rate
        # model.ent_coef = args.ent_coef
        # # model.clip_range = args.clip_range
        # model.n_steps = args.n_steps
        # new_logger = configure(save_path, ["tensorboard"])
        # model.set_logger(new_logger)
    
    return model