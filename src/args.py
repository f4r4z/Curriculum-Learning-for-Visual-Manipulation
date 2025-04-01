# This module contains commandline args that might be shared between different scripts
from dataclasses import dataclass
from typing import List, Optional, Union, Type

import os
import torch
import numpy as np
import h5py
import pickle
import wandb

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3 import PPO, SAC

from rllte.xplore.reward import RND, Disagreement, E3B, Fabric, ICM, NGU, PseudoCounts, RE3, RIDE
from rllte.common.prototype import BaseReward

from .callbacks import RLeXploreWithOffPolicyRL, RLeXploreWithOnPolicyRL


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
    
    @property
    def alg_class(self):
        if self.alg == 'ppo':
            return PPO
        elif self.alg == 'sac':
            return SAC
        raise Exception(f"invalid alg '{self.alg}'")
    
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
    shaping_reward: bool = True
    """if toggled, shaping reward will be off for all goal states"""
    sparse_reward: float = 10.0
    """total sparse reward for success"""
    reward_geoms: Optional[str] = None
    """if geoms are passed, those specific geoms will be rewarded, for single object predicates only [format example: ketchup_1_g1,ketchup_1_g2]"""
    dense_reward_multiplier: float = 1.0
    """multiplies the last goal state's shaping reward"""
    steps_per_episode: int = 250
    """number of steps in episode. If truncate is True, the episode will terminate after this value"""
    setup_demo_path: Optional[str] = None
    """a directory containing a demo.hdf5 file. If passed in, runs the actions in the given demonstration before every episode to setup the scene"""
    sim_states_path: Optional[str] = None
    """path to initial sim states pickle file in order to randomly select initial sim states. if None, the sim state will always start from default"""

    def fetch_sim_states(self):
        """load and cache the sim states from sim_states_path. If cached, directly return the states and do not load"""
        if self.sim_states_path is None:
            return None
        if not hasattr(self, 'sim_states'):
            self.sim_states: Optional[np.ndarray] = None
        if self.sim_states is None:
            with open(self.sim_states_path, "rb") as f:
                self.sim_states = pickle.load(f)
        return self.sim_states
    
    def fetch_setup_demo(self):
        if self.setup_demo_path is None:
            return None
        if not hasattr(self, 'setup_demo'):
            self.setup_demo: Optional[np.ndarray] = None
        if self.setup_demo is None:
            hdf5_path = os.path.join(self.setup_demo_path, "demo.hdf5")
            f = h5py.File(hdf5_path, "r")
            self.setup_demo = f['data/demo_1/actions'][:]
            if self.verbose >= 1: print("loaded setup demo with length", len(self.setup_demo))
        return self.setup_demo

    

@dataclass
class WandbArgs:
    wandb_project: str = "cl_manipulation"
    """wandb project name"""
    wandb_entity: str = "<YOUR_WANDB_ENTITY>"
    """wandb entity name (username)"""
    wandb: bool = False
    """if toggled, model will log to wandb otherwise it would not"""

    def init_wandb_if_toggled(self, **wandb_args):
        if self.wandb:
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                **wandb_args
            )