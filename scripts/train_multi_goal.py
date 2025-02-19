# add parent path to sys
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
from dataclasses import dataclass, field
import tyro
import imageio
from IPython.display import HTML
import wandb
import torch
import numpy as np
import typing

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO, SAC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.logger import configure

from src.her_replay_buffer_modified import HerReplayBufferModified
from src.envs_gymapi import LowDimensionalObsGymEnv, LowDimensionalObsGymGoalEnv, AgentViewGymEnv, AgentViewGymGoalEnv
from src.networks import CustomCNN, CustomCombinedExtractor, CustomCombinedExtractor2, CustomCombinedPatchExtractor
from src.callbacks import TensorboardCallback, RLeXploreWithOffPolicyRL, RLeXploreWithOnPolicyRL, VideoWriter

from rllte.xplore.reward import RND, Disagreement, E3B, Fabric, ICM, NGU, PseudoCounts, RE3, RIDE

@dataclass
class Args:
    # User specific arguments
    seed: int = None
    """random seed for reproducibility"""
    video_path: str = "videos/output.mp4"
    """file path of the video output file"""
    save_path: str = "event_logs"
    """file path of the model output file"""
    load_path: str = None # "models/checkpoints/"
    """directory path of the models checkpoints"""
    model_path: str = None
    """path to existing model if loading a model"""
    policy_1_path: str = None
    """in case of multi-goal states, pass in the path for policy for first goal"""
    wandb_project: str = "cl_manipulation"
    """wandb project name"""
    wandb_entity: str = "<YOUR_WANDB_ENTITY>"
    """wandb entity name (username)"""
    wandb: bool = False
    """if toggled, model will log to wandb otherwise it would not"""

    # Environment specific arguments
    custom_bddl_path: str = None
    """if passed in, the custom path will be used for bddl file as opposed to libero default files"""
    bddl_file_name: str = "libero_90/KITCHEN_SCENE6_close_the_microwave.bddl"
    """file name of the BDDL file"""
    visual_observation: bool = False
    """if toggled, the environment will return visual observation otherwise it would not"""
    shaping_reward: bool = True
    """if toggled, shaping reward will be off for all goal states"""
    sparse_reward: float = 10.0
    """total sparse reward for success"""
    reward_geoms: str = None
    """if geoms are passed, those specific geoms will be rewarded, for single object predicates only [format example: ketchup_1_g1,ketchup_1_g2]"""
    dense_reward_multiplier: float = 1.0
    """multiplies the last goal state's shaping reward"""
    steps_per_episode: int = 250
    """number of steps in episode. If truncate is True, the episode will terminate after this value"""

    # Algorithm specific arguments
    alg: str = "ppo"
    """algorithm to use for training: ppo, sac"""
    her: bool = False
    """if toggled, SAC will use HER otherwise it would not"""
    exploration_alg: str = None
    """algorithm for exploration techniques: rnd, e3b, disagreement, re3, ride, icm"""
    total_timesteps: int = 250000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3 
    """the learning rate of the optimizer"""
    n_steps: int = 512
    """number of steps to run for each environment per update"""
    num_envs: int = 1
    """number of LIBERO environments"""
    ent_coef: float = 0.0
    """entropy coefficient for the loss calculation"""
    clip_range: float = 0.2
    """Clipping parameter, it can be a function of the current progress remaining (from 1 to 0)."""
    truncate: bool = True
    """if toggled, algorithm with truncate after 250 steps"""
    progress_bar: bool = True
    """if toggled, progress bar will be shown"""
    device: str = ""
    """device to use for training"""

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
    
    # if args.shaping_reward:
    #     env_args["shaping_reward"] = True
    
    # set up reward geoms
    reward_geoms = args.reward_geoms.split(",") if args.reward_geoms is not None else None
    # pass in policy 1
    policy_1 = PPO.load(f"{args.policy_1_path}", device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    if not args.truncate:
        env_args["horizon"] = args.total_timesteps

    print("Setting up environment")
    vec_env_class = SubprocVecEnv if args.num_envs > 1 else DummyVecEnv
    if args.visual_observation:
        if args.her:
            envs = vec_env_class(
                [lambda: Monitor(AgentViewGymGoalEnv(**env_args), info_keywords=["is_success"]) for _ in range(args.num_envs)]
            )
        else:
            envs = vec_env_class(
                [lambda: Monitor(AgentViewGymEnv(**env_args), info_keywords=["is_success"]) for _ in range(args.num_envs)]
            )
    else:
        if args.her:
            envs = vec_env_class(
                [lambda: Monitor(LowDimensionalObsGymGoalEnv(**env_args), info_keywords=["is_success"]) for _ in range(args.num_envs)]
            )
        else:
            envs = vec_env_class(
                [lambda: Monitor(LowDimensionalObsGymEnv(args.shaping_reward, args.sparse_reward, reward_geoms, args.dense_reward_multiplier, args.steps_per_episode, goal_1_policy=policy_1, **env_args), info_keywords=["is_success"]) for _ in range(args.num_envs)]
            )


    # Seeding everything
    if args.seed is not None:
        envs.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    print("Start training")
    alg_str = args.alg
    if args.her:
        alg_str = f"her_{args.alg}"
    if args.exploration_alg is not None:
        alg_str = f"{args.exploration_alg}_{alg_str}"
    alg_str = f"{alg_str}_seed_{args.seed}"
    run_name = os.path.join(task_name, alg_str, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
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
            envs,
            verbose=1,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            tensorboard_log=save_path,
            n_steps=args.n_steps,
            ent_coef=args.ent_coef,
            # clip_range=args.clip_range,
            seed=args.seed
        )
        log_interval = 1
    elif args.alg == "sac":
        algorithm = SAC
        if args.her:
            model = SAC(
                policy_class,
                envs,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=save_path,
                seed=args.seed,
                learning_rate=args.learning_rate,
                learning_starts=1000*args.num_envs,
                batch_size=256,
                train_freq=(1, "step"),
                gradient_steps=-1,
                replay_buffer_class=HerReplayBufferModified,
                replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future',)
            )
        else:
            model = SAC(
                policy_class,
                envs,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=save_path,
                seed=args.seed,
                learning_rate=args.learning_rate,
                learning_starts=1000*args.num_envs,
                batch_size=256,
                train_freq=(1, "step"),
                gradient_steps=-1,
            )
        log_interval = 2
    else:
        raise ValueError(f"Algorithm {args.alg} is not in supported list [ppo, sac]")

    if args.model_path:
        print("loading model from ", args.model_path)
        model = algorithm.load(f"{args.model_path}", env=envs)
        model.learning_rate = args.learning_rate
        model.ent_coef = args.ent_coef
        # model.clip_range = args.clip_range
        model.n_steps = args.n_steps
        new_logger = configure(save_path, ["tensorboard"])
        model.set_logger(new_logger)
    
    # get device
    print("devices: ", [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())])
    if not args.device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    '''
    callback list
    '''
    callbacks = []

    # checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=log_interval*1024, save_path=save_path, name_prefix="model")
    callbacks.append(checkpoint_callback)

    # log videos
    callbacks.append(VideoWriter(n_steps=5000 * args.num_envs))

    # exploration technique callbacks
    if args.exploration_alg is not None:
        if args.exploration_alg == "rnd":
            irs = RND(envs, device=device, rwd_norm_type='none', obs_norm_type='none')
        elif args.exploration_alg == "disagreement":
            irs = Disagreement(envs, device=device, rwd_norm_type='none', obs_norm_type='none')
        elif args.exploration_alg == "e3b":
            irs = E3B(envs, device=device, rwd_norm_type='none', obs_norm_type='none')
        elif args.exploration_alg == "icm":
            irs = ICM(envs, device=device, rwd_norm_type='none', obs_norm_type='none')
        elif args.exploration_alg == "pseudocounts":
            irs = PseudoCounts(envs, device=device, rwd_norm_type='rms', obs_norm_type='none')
        elif args.exploration_alg == "ride":
            irs = RIDE(envs, device=device, rwd_norm_type='rms', obs_norm_type='none')
        else:
            print("You have typed an invalid exploration technique. --help for more information.")
            exit(1)
        # on policy vs off policy
        if args.alg == 'ppo':
            callbacks.append(RLeXploreWithOnPolicyRL(irs))
        else:
            callbacks.append(RLeXploreWithOffPolicyRL(irs))
    
    '''train'''
    model.learn(total_timesteps=args.total_timesteps, log_interval=log_interval, callback=callbacks, progress_bar=False)
    model.save(save_path)

    del model