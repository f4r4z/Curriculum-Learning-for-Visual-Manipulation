from dataclasses import dataclass
from typing import List, Optional, Union

import imageio
from IPython.display import HTML

import torch

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, SAC

from rllte.xplore.reward import RND, Disagreement, E3B, Fabric, ICM, NGU, PseudoCounts, RE3, RIDE
from rllte.common.prototype import BaseReward

from src.envs_gymapi import LowDimensionalObsGymEnv, LowDimensionalObsGymGoalEnv, AgentViewGymEnv, AgentViewGymGoalEnv
from src.networks import CustomCNN, CustomCombinedPatchExtractor
from src.her_replay_buffer_modified import HerReplayBufferModified
from src.callbacks import RLeXploreWithOffPolicyRL, RLeXploreWithOnPolicyRL


@dataclass
class AlgArgs:
    alg: str = "ppo"
    """algorithm to use for training: ppo, sac"""
    visual_observation: bool = False
    """if toggled, the environment will return visual observation otherwise it would not"""
    her: bool = False
    """if toggled, SAC will use HER otherwise it would not"""
    exploration_alg: Optional[str] = None
    """algorithm for exploration techniques: rnd, e3b, disagreement, re3, ride, icm"""
    total_timesteps: int = 250000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3 
    """the learning rate of the optimizer"""
    n_steps: int = 512
    """number of steps to run for each environment per update"""
    ent_coef: float = 0.0
    """entropy coefficient for the loss calculation"""
    clip_range: float = 0.2
    """Clipping parameter, it can be a function of the current progress remaining (from 1 to 0)."""
    truncate: bool = True
    """if toggled, algorithm with truncate after 250 steps"""
    progress_bar: bool = True
    """if toggled, progress bar will be shown"""
    device: Optional[str] = None
    """device to use for training"""
    
    def get_alg_str(self):
        alg_str = self.alg
        if self.her:
            alg_str = f"her_{alg_str}"
        if self.exploration_alg is not None:
            alg_str = f"{self.exploration_alg}_{alg_str}"
        return alg_str
    
    def get_device(self):
        print("devices: ", [torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())])
        if not self.device:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(self.device)
    
    def get_exploration_alg_reward(self, envs: VecEnv, device: Optional[torch.device] = None) -> Optional[BaseReward]:
        if device == None:
            device = self.get_device()
        if self.exploration_alg == None:
            return None
        elif self.exploration_alg == "rnd":
            return RND(envs, device=device, rwd_norm_type='none', obs_norm_type='none')
        elif self.exploration_alg == "disagreement":
            return Disagreement(envs, device=device, rwd_norm_type='none', obs_norm_type='none')
        elif self.exploration_alg == "e3b":
            return E3B(envs, device=device, rwd_norm_type='none', obs_norm_type='none')
        elif self.exploration_alg == "icm":
            return ICM(envs, device=device, rwd_norm_type='none', obs_norm_type='none')
        elif self.exploration_alg == "pseudocounts":
            return PseudoCounts(envs, device=device, rwd_norm_type='rms', obs_norm_type='none')
        elif self.exploration_alg == "ride":
            return RIDE(envs, device=device, rwd_norm_type='rms', obs_norm_type='none')
        else:
            print("You have typed an invalid exploration technique. --help for more information.")
            exit(1)
    
    def get_exploration_callback(self, envs: VecEnv, device: Optional[torch.device] = None):
        irs = self.get_exploration_alg_reward(envs, device)
        if irs == None:
            return None
        # on policy vs off policy
        if self.alg == 'ppo':
            return RLeXploreWithOnPolicyRL(irs)
        else:
            return RLeXploreWithOffPolicyRL(irs)

@dataclass
class EnvArgs:
    """Args used when initializing the environments"""
    num_envs: int = 1
    """number of LIBERO environments"""
    multiprocessing_start_method: Optional[str] = None
    """The start method for starting processes if num_envs > 1. Can be 'fork', 'spawn', or 'forkserver'. 'forkserver' is default"""
    setup_demo_path: Optional[str] = None
    """If passed in, runs the actions in the given demonstration before every episode to setup the scene"""
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
    

@dataclass
class WandbArgs:
    wandb_project: str = "cl_manipulation"
    """wandb project name"""
    wandb_entity: str = "<YOUR_WANDB_ENTITY>"
    """wandb entity name (username)"""
    wandb: bool = False
    """if toggled, model will log to wandb otherwise it would not"""


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
    

def setup_envs(
    bddl_file: str,
    args: Union[EnvArgs, AlgArgs],
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
                setup_demo=args.setup_demo_path,
                **env_args
            ), info_keywords=["is_success"]) for _ in range(args.num_envs)]
    
    if args.num_envs > 1:
        print("method", args.multiprocessing_start_method)
        return SubprocVecEnv(envs, start_method=args.multiprocessing_start_method)
    else:
        return DummyVecEnv(envs)


def setup_model(
    args: AlgArgs, 
    env: VecEnv, seed, 
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