from typing import List, Optional, Union
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from stable_baselines3.common.vec_env import VecEnv
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv


class AgentViewEnv(OffScreenRenderEnv):
    """ Sparse reward environment with only the agent view as observation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        agent_view_shape = self.env._get_observations()["agentview_image"].shape
        self.observation_space = Box(low=0, high=255, shape=agent_view_shape, dtype="uint8")
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype="float32")
        self.step_count = 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        success = self.check_success()
        reward = 10 * success
        self.step_count += 1
        done = success or self.step_count > 500
        return obs["agentview_image"], reward, done, info

    def reset(self):
        obs = super().reset()
        self.step_count = 0
        return obs["agentview_image"]


class LowDimensionalObsEnv(OffScreenRenderEnv):
    """ Sparse reward environment with all the low-dimensional states
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        obs = self.env._get_observations()
        low_dim_obs = self.get_low_dim_obs(obs)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=low_dim_obs.shape, dtype="float32")
        self.action_space = Box(low=-1, high=1, shape=(7,), dtype="float32")
        self.step_count = 0
    
    def get_low_dim_obs(self, obs):
        return np.concatenate([
            obs[k] for k in obs.keys() if not k.endswith("image")
        ], axis = -1)
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        success = self.check_success()
        reward = 10 * success
        self.step_count += 1
        done = success or self.step_count > 500
        return self.get_low_dim_obs(obs), reward, done, info
    
    def reset(self):
        obs = super().reset()
        self.step_count = 0
        return self.get_low_dim_obs(obs)
    

class GymVecEnvs(VecEnv):
    """ Vectorized environment for gymnasium environments
    """
    def __init__(self, envs: SubprocVectorEnv):
        self.envs = envs
        num_envs = len(envs)
        super().__init__(num_envs, envs.observation_space[0], envs.action_space[0])
        self.rewards = np.zeros(num_envs)

    def reset(self):
        return self.envs.reset()
    
    def step(self, actions):
        obs, rewards, dones, infos = self.envs.step(actions)
        id, *_ = np.where(dones)
        if len(id) > 0:
            obs_new = self.envs.reset(id=id)
            obs[id] = obs_new
        self.rewards = rewards
        return obs, rewards, dones, infos
    
    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs, rewards, dones, infos = self.envs.step(self.actions)
        id, *_ = np.where(dones)
        if len(id) > 0:
            obs_new = self.envs.reset(id=id)
            obs[id] = obs_new
        self.rewards = rewards
        return obs, rewards, dones, infos

    def close(self) -> None:
        return self.envs.close()
    
    def env_is_wrapped(self, wrapper_class, indices):
        return False
    
    def get_attr(self, attr_name, indices = None):
        return self.envs.get_env_attr(attr_name)
    
    def set_attr(self, attr_name, value, indices):
        self.envs.set_env_attr(attr_name, value, indices)

    def env_method(self, method_name: str, *method_args, indices: Optional[Union[int, List[int], np.ndarray]] = None, **method_kwargs):
        return super().env_method(method_name, *method_args, indices=indices, **method_kwargs)
